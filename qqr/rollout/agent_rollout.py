import asyncio
import copy
import inspect
import json
import logging
import uuid
from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from pprint import pformat
from typing import Any

import anyio
import numpy as np
from slime.backends.sglang_utils.server_control import abort_servers_until_idle
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import (
    MetricGatherer,
    call_dynamic_filter,
    should_drop_dynamic_filter_output,
)
from slime.rollout.rm_hub import async_rm, batched_async_rm
from slime.rollout.sample_hooks import apply_rollout_sample_hooks
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, get_rollout_num_engines, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.trace_utils import (
    build_sglang_meta_trace_attrs,
    trace_function,
    trace_span,
)
from tqdm.auto import tqdm

from qqr.mcp import MCPServer, MCPServerManager
from qqr.mcp.utils import get_mcp_tools
from qqr.schemas import Sample
from qqr.utils.envs import RETRY_STOP_AFTER_ATTEMPT, RETRY_WAIT_FIXED

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)

_PROCESSOR_PROMPT_KEYS = {"input_ids", "attention_mask"}


def _prepare_prompt_ids(sample: Sample, tokenizer, processor: Any) -> list[int]:
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal_inputs = any(
        value is not None for value in raw_multimodal_inputs.values()
    )

    tools = sample.train_metadata.get("tools") if sample.train_metadata else None
    prompt_text = tokenizer.apply_chat_template(
        sample.messages, tools=tools, tokenize=False, add_generation_prompt=True
    )

    if processor and has_multimodal_inputs:
        processor_output = processor(
            text=prompt_text, **build_processor_kwargs(raw_multimodal_inputs)
        )
        prompt_ids = processor_output["input_ids"][0]
        if sample.multimodal_train_inputs is None:
            sample.multimodal_train_inputs = {
                k: v
                for k, v in processor_output.items()
                if k not in _PROCESSOR_PROMPT_KEYS
            } or None
        return prompt_ids

    return tokenizer.encode(prompt_text, add_special_tokens=False)


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * get_rollout_num_engines(args)
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=self.tokenizer.eos_token,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )
        if args.rollout_top_p != 1.0:
            self.sampling_params["custom_params"] = {"return_top_p_token_ids": True}

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [
                sampling_seed_base + i for i in range(args.n_samples_per_prompt)
            ]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [
            i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)
        ]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


class MCPState(metaclass=SingletonMeta):
    """
    The global state for the MCP server.
    """

    def __init__(
        self,
        manager: MCPServerManager,
        max_retry_attempts: int = RETRY_STOP_AFTER_ATTEMPT,
        retry_backoff_seconds_base: float = RETRY_WAIT_FIXED,
    ):
        """
        Args:
            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to 3 retries.
            retry_backoff_seconds_base: The base delay, in seconds, used for exponential
                backoff between retries.
        """

        self._server_lock = asyncio.Lock()

        self.servers: list[MCPServer] = None
        self.server_versions = defaultdict(int)
        self.manager = manager

        self.tools = []
        self.tool_to_server: dict[str, MCPServer] = {}

        self.max_retry_attempts = max_retry_attempts
        self.retry_backoff_seconds_base = retry_backoff_seconds_base

    async def get_servers(self) -> list[MCPServer]:
        """
        Thread-safe lazy initialization of the MCP server.
        """
        # Double-checked locking pattern for async init
        if self.servers is None:
            async with self._server_lock:
                if self.servers is not None:
                    return self.servers

                try:
                    await self.manager.connect_all()
                    for server in self.manager.active_servers:
                        converted_tools = await get_mcp_tools(server)
                        self.tools += converted_tools
                        for tool in converted_tools:
                            self.tool_to_server[tool["function"]["name"]] = server

                        self.server_versions[server.name] += 1
                        logger.info(f"MCP Server {server.name} connected successfully.")

                    self.servers = self.manager.active_servers

                except Exception as e:
                    logger.error(f"Failed to initialize MCP Servers: {e}")
                    self.servers = None
                    raise e

        return self.servers

    async def call_tool(self, tool_call: dict) -> dict:
        await self.get_servers()

        tool_name = tool_call["function"]["name"]
        tool_call_id = tool_call["id"]
        tool_content = ""

        target_server = self.tool_to_server.get(tool_name)

        if not target_server:
            return {
                "role": "tool",
                "content": f"[Error] Tool '{tool_name}' not found in any connected MCP servers.",
                "tool_call_id": tool_call_id,
            }

        tool_arguments_str = tool_call["function"]["arguments"]
        try:
            tool_arguments = (
                json.loads(tool_arguments_str) if tool_arguments_str else {}
            )
        except json.JSONDecodeError as e:
            return {
                "role": "tool",
                "content": f"[Error] Invalid JSON arguments: {e}",
                "tool_call_id": tool_call_id,
            }

        for attempts in range(self.max_retry_attempts):
            server_version = self.server_versions[target_server.name]
            should_reconnect = False

            try:
                result = await target_server.call_tool(tool_name, tool_arguments)
                tool_content = "\n\n---\n\n".join(
                    item.text for item in result.content if item.type == "text"
                )
                break

            except anyio.ClosedResourceError as e:
                logger.warning(f"[MCP] '{target_server.name}' connection lost: {e}")
                tool_content = f"[Error] Tool connection lost: {e}"
                should_reconnect = True

            except Exception as e:
                tool_content = f"[Error] Tool execution failed: {e}"

            if attempts + 1 >= self.max_retry_attempts:
                break

            if should_reconnect:
                async with self._server_lock:
                    if self.server_versions[target_server.name] > server_version:
                        continue

                    logger.warning(
                        f"[MCP] '{target_server.name}' ({server_version}) retrying {attempts + 1}/{self.max_retry_attempts}..."
                    )
                    await self.manager._cleanup_server(target_server)
                    await self.manager._attempt_connect(target_server)
                    self.manager._refresh_active_servers()

                    self.server_versions[target_server.name] += 1

            backoff = self.retry_backoff_seconds_base * (2**attempts)
            await asyncio.sleep(backoff)

        return {
            "role": "tool",
            "content": tool_content,
            "tool_call_id": tool_call_id,
        }


async def generate(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any]
) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)

    current_sampling_params = deepcopy(sampling_params)
    current_sampling_params["max_new_tokens"] = min(
        sampling_params["max_new_tokens"],
        args.rollout_max_context_len - len(prompt_ids),
    )

    if current_sampling_params["max_new_tokens"] <= 100:
        sample.response = ""
        sample.tokens = [state.tokenizer.pad_token_id]
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.reward = 0.0
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Enforce greedy decoding for the first response.
    # Could enable deterministic inference.
    if args.n_samples_per_prompt > 1 and sample.index % args.n_samples_per_prompt == 0:
        current_sampling_params["temperature"] = 0.0
        current_sampling_params["top_k"] = 1

    # Prepare payload for sglang server
    payload = {
        "sampling_params": current_sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    images = (
        sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
    )
    if images:
        payload["image_data"] = [
            encode_image_for_rollout_engine(image) for image in images
        ]
        # For single-turn multimodal requests, send text so SGLang expands the
        # image placeholders with its own processor rules.
        payload["text"] = sample.prompt
    else:
        payload["input_ids"] = prompt_ids

    if not sample.tokens:
        sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing (SGLang Model Gateway)
    headers = None
    if sample.session_id:
        if getattr(args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": sample.session_id}

    with trace_span(
        sample,
        "sglang_generate",
        attrs={"max_new_tokens": sampling_params["max_new_tokens"]},
    ) as span:
        output = await post(url, payload, headers=headers)
        span.update(build_sglang_meta_trace_attrs(output["meta_info"]))

    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [
            item[1] for item in output["meta_info"]["output_token_logprobs"]
        ]
        new_response_log_probs = [
            item[0] for item in output["meta_info"]["output_token_logprobs"]
        ]
    else:
        new_response_tokens, new_response_log_probs = [], []

    # Keep the agent overwrite (`=`) semantics: each turn re-derives the prompt
    # from messages, so the response-side fields must describe only this turn's
    # output. Reset them before append_response_tokens, which appends.
    if sample.loss_mask is not None:
        assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None

    sample.append_response_tokens(
        args,
        tokens=new_response_tokens,
        log_probs=new_response_log_probs,
        trainable=True,
        meta_info=output["meta_info"],
        text=output["text"],
    )

    return sample


@trace_function("generate_and_rm", target="sample")
async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if (
        args.partial_rollout
        and args.mask_offpolicy_in_partial_rollout
        and sample.response_length > 0
    ):
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if (
        sample.status == Sample.Status.COMPLETED
        or sample.status == Sample.Status.TRUNCATED
    ):
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = (
                getattr(sample, "generate_function_path", None)
                or args.custom_generate_function_path
            )

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(
                        args, sample, sampling_params, evaluation=evaluation
                    )
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    sample = await apply_rollout_sample_hooks(args, sample, evaluation=evaluation)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    if isinstance(sample, list):
        samples = sample
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            return samples

        samples_need_reward = [sample for sample in samples if sample.reward is None]
        with trace_span(samples_need_reward, "reward_model"):
            rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # Some custom generate paths may have already filled the reward.
        if sample.reward is None:
            with trace_span(sample, "reward_model"):
                sample.reward = await async_rm(args, sample)

    return sample


@trace_function(
    "generate_and_rm_group",
    target="group",
    attrs_getter=lambda args, group, sampling_params, evaluation=False: {
        "group_size": len(group)
    },
)
async def generate_and_rm_group(
    args: Namespace,
    group: list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample] | list[list[Sample]]:
    # ``generate_and_rm`` may return either a ``Sample`` or a ``list[Sample]``
    # depending on whether the ``--custom-generate-function-path`` callable
    # emits one trainable sample or several (e.g. multi-turn agent rollouts
    # that fan out into multiple prefix-chained samples). The asyncio.gather
    # below preserves whichever shape each task produced, so the group is
    # ``list[Sample]`` for plain rollouts and ``list[list[Sample]]`` for
    # the fan-out case.
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(
                generate_and_rm(
                    args, sample, current_sampling_params, evaluation=evaluation
                )
            )
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        with trace_span(group, "group_reward_model"):
            rewards = await batched_async_rm(args, group)

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    response = await get(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers"
    )
    urls = [worker["url"] for worker in response["workers"]]

    await abort_servers_until_idle(urls)

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(
            state.pendings, return_when=asyncio.FIRST_COMPLETED
        )

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if args.dynamic_sampling_filter_path is not None
        else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(
        total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation"
    )
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            if isinstance(samples[0], list):
                samples = [[Sample(**_s.__dict__.copy()) for _s in s] for s in samples]
            else:
                samples = [Sample(**s.__dict__.copy()) for s in samples]
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(
            state.pendings, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][-1] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample:\nmessages:\n{pformat(sample.messages, sort_dicts=False)}\nreward: {sample.reward}"
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if should_drop_dynamic_filter_output(
                dynamic_filter_output,
                remaining_batch_size=state.remaining_batch_size,
                target_data_size=target_data_size,
            ):
                metric_gatherer.on_dynamic_filter_drop(
                    reason=dynamic_filter_output.reason
                )
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][-1] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout:\nmessages:\n{pformat(sample.messages, sort_dicts=False)}\nreward: {sample.reward}"
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, (
        f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    )
    data = sorted(
        data,
        key=lambda group: (
            group[0][0].index if isinstance(group[0], list) else group[0].index
        ),
    )
    all_samples = sorted(
        all_data,
        key=lambda group: (
            group[0][0].index if isinstance(group[0], list) else group[0].index
        ),
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(
        samples=data, metrics=metric_gatherer.collect()
    ), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(
    args: Namespace, rollout_id: int
) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    global EVAL_PROMPT_DATASET

    eval_multimodal_keys = (
        dataset_cfg.multimodal_keys
        if dataset_cfg.multimodal_keys is not None
        else args.multimodal_keys
    )
    eval_apply_chat_template = (
        dataset_cfg.apply_chat_template
        if dataset_cfg.apply_chat_template is not None
        else args.apply_chat_template
    )
    eval_apply_chat_template_kwargs = (
        dataset_cfg.apply_chat_template_kwargs
        if dataset_cfg.apply_chat_template_kwargs is not None
        else args.apply_chat_template_kwargs
    )

    cache_key = dataset_cfg.cache_key + (
        args.hf_checkpoint,
        eval_apply_chat_template,
        json.dumps(eval_multimodal_keys, sort_keys=True)
        if eval_multimodal_keys is not None
        else None,
        (
            json.dumps(eval_apply_chat_template_kwargs, sort_keys=True)
            if eval_apply_chat_template_kwargs is not None
            else None
        ),
    )
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=eval_multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=eval_apply_chat_template,
            apply_chat_template_kwargs=eval_apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=(
            dataset_cfg.skip_special_tokens
            if dataset_cfg.skip_special_tokens is not None
            else args.rollout_skip_special_tokens
        ),
        no_stop_trim=dataset_cfg.no_stop_trim
        if dataset_cfg.no_stop_trim is not None
        else True,
        spaces_between_special_tokens=False,
    )
    if dataset_cfg.repetition_penalty is not None:
        base_sampling_params["repetition_penalty"] = dataset_cfg.repetition_penalty

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(
                getattr(sample, "metadata", None)
            )
            sample.custom_rm_path = dataset_cfg.custom_rm_path
            sample.generate_function_path = getattr(
                dataset_cfg, "custom_generate_function_path", None
            )
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logged_sample = sample[0] if isinstance(sample, list) else sample
            logger.info(
                f"eval_rollout_single_dataset example data:\n"
                f"messages:\n{pformat(logged_sample.messages, sort_dicts=False)}\n"
                f"reward: {logged_sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [
                sample.reward if not reward_key else sample.reward[reward_key]
                for sample in data
            ],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(
        generate_rollout_async(args, rollout_id, data_source.get_samples)
    )
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
