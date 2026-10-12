# qqr/off_policy/rti/ref/ref_scorer.py
import asyncio
import logging
import math
from argparse import Namespace
from typing import Any

from slime.utils.http_utils import post
from slime.utils.misc import SingletonMeta

from qqr.rollout.agent_rollout import GenerateState
from qqr.schemas import Sample

logger = logging.getLogger(__name__)


class RefScorer(metaclass=SingletonMeta):
    """
    Teacher-forcing scorer for log P(completion_text | messages).

    - Reuses GenerateState(args) for tokenizer/processor/chat_template
    - Uses the SAME router host/port as rollout, but calls OpenAI-compat endpoint:
        POST http://{router}/v1/completions
      with echo=True + logprobs to get per-token logprobs of the provided sequence.

    This yields true conditional probability of the provided completion_text
    (assuming router implements OpenAI echo logprobs correctly).
    """

    def __init__(self, args: Namespace, url=None, model_id=None) -> None:
        self.args = args
        conc = int(getattr(args, "ref_scorer_concurrency", 64))
        self._sem = asyncio.Semaphore(conc)

        if url is None or model_id is None:
            base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}".rstrip("/")
            self.url = f"{base_url}/v1/completions"
            self.generate_url = f"{base_url}/generate"
            self.model_id = getattr(args, "ref_model_id", None) or getattr(args, "model", None) or getattr(args, "hf_checkpoint", None)
            if not self.model_id:
                raise RuntimeError("Missing model id. Set args.ref_model_id or args.model or args.hf_checkpoint.")
        else:
            self.url = url
            # Derive generate URL: strip known suffixes and append /generate
            base = url.rstrip("/")
            for suffix in ("/v1/completions", "/v1/chat/completions", "/generate"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            self.generate_url = f"{base}/generate"
            self.model_id = model_id


    def _ppl_from_sumlogp(self, sum_logp: float, n_tokens: int) -> float:
        if n_tokens <= 0 or sum_logp == float("-inf") or math.isnan(sum_logp):
            return float("inf")
        return float(math.exp(-sum_logp / n_tokens))

    def _find_last_chat_response_start(self, tokens: list[str]) -> int | None:
        """
        Heuristic: find last <|im_start|> then skip role + newline.
        This is model/template dependent. If your template differs, adjust here.
        """
        # For many Qwen-style templates, tokens include "<|im_start|>"
        # followed by "assistant" then "\n".
        idxs = [i for i, t in enumerate(tokens) if t == "<|im_start|>"]
        if not idxs:
            return None
        i = idxs[-1]
        # best-effort skip: <|im_start|> assistant \n
        return i

    async def score_sample_completion_async(
        self,
        sample: Sample,
        completion_text: str,
        *,
        max_context_length: int | None = None,
    ) -> dict[str, Any]:
        """
        Returns logprobs for tokens in completion_text under the model conditioned on sample.messages.

        Uses sglang's native /generate endpoint with return_logprob=True and max_new_tokens=0,
        which returns per-token logprobs as (logprob, token_id, ...) tuples in
        meta_info["input_token_logprobs"]. This avoids the broken string-token splitting
        that occurs with /v1/completions echo=True in newer transformers/sglang versions.
        """
        args = self.args
        state = GenerateState(args)

        tools = sample.train_metadata.get("tools") if sample.train_metadata else None

        prompt_text = state.tokenizer.apply_chat_template(
            sample.messages, tools=tools, tokenize=False, add_generation_prompt=True
        )

        if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
            raise NotImplementedError("Teacher-forcing scoring does not support multimodal images here.")

        full_text = prompt_text + completion_text

        mcl = int(max_context_length or getattr(args, "rollout_max_context_len", 8192))

        # Compute token counts using the local tokenizer so we can slice the completion part.
        prompt_ids = state.tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids = state.tokenizer.encode(completion_text, add_special_tokens=False)
        full_ids = state.tokenizer.encode(full_text, add_special_tokens=False)
        comp_len = len(full_ids) - len(prompt_ids)
        if comp_len < 0:
            comp_len = 0

        # Use sglang native /generate endpoint with max_new_tokens=0 to score the full sequence.
        # input_token_logprobs: list of (logprob, token_id, ...) for every input token.
        # The first entry has no predecessor, so its logprob is typically None/0; we skip it with [1:].
        payload = {
            "text": full_text,
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
            },
            "return_logprob": True,
            "logprob_start_len": 0,  # return logprobs for all input tokens
        }

        async with self._sem:
            out = await post(self.generate_url, payload)

        meta = out.get("meta_info", {}) or {}
        input_token_logprobs = meta.get("input_token_logprobs", None)
        if input_token_logprobs is None:
            raise RuntimeError(
                f"sglang /generate did not return input_token_logprobs. "
                f"meta_info keys={list(meta.keys())}. "
                f"Ensure return_logprob=True is supported by your sglang version."
            )

        # input_token_logprobs is a list of (logprob, token_id, ...) tuples.
        # Entry [0] corresponds to the first token which has no logprob (skip it).
        # We take the last comp_len entries for the completion part.
        # Total entries after [1:] = len(full_ids) - 1, and we want the last comp_len of those.
        all_lp_entries = input_token_logprobs[1:]  # skip the first token (no logprob)

        if comp_len > 0:
            comp_entries = all_lp_entries[-comp_len:]
        else:
            comp_entries = []

        # Extract (logprob, token_id) from each entry tuple
        comp_lps_float = []
        comp_token_ids_from_server = []
        for entry in comp_entries:
            lp = entry[0] if entry[0] is not None else float("-inf")
            tid = entry[1] if len(entry) > 1 else None
            comp_lps_float.append(float(lp))
            if tid is not None:
                comp_token_ids_from_server.append(int(tid))

        # Prefer token ids from local tokenizer encoding for consistency
        result_token_ids = completion_ids if completion_ids else comp_token_ids_from_server

        sum_logp = float(sum(comp_lps_float)) if comp_lps_float else float("-inf")
        avg_logp = sum_logp / max(1, len(comp_lps_float)) if comp_lps_float else float("-inf")
        ppl = self._ppl_from_sumlogp(sum_logp, len(comp_lps_float))

        return {
            "mode": "teacher_forcing_sglang_generate_return_logprob",
            "prompt_len": len(prompt_ids),
            "n_tokens": len(comp_lps_float),
            "tokens": result_token_ids,
            "token_logprobs": comp_lps_float,
            "sum_logp": sum_logp,
            "avg_logp": avg_logp,
            "ppl": ppl,
            "max_context_length": mcl,
            "model": self.model_id,
        }

    async def score_completion_async(
        self,
        messages: list[dict[str, Any]],
        completion_text: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_context_length: int | None = None,
        sample_index: int = 0,
    ) -> dict[str, Any]:
        s = Sample(
            index=sample_index,
            group_index=0,
            messages=messages,
            response="",
            status=Sample.Status.PENDING,
            train_metadata={"tools": tools} if tools else None,
            multimodal_inputs={},
        )
        return await self.score_sample_completion_async(
            s, completion_text, max_context_length=max_context_length
        )

    def score_completion(self, *args, **kwargs) -> dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("Use await score_completion_async() inside an event loop.")
        return asyncio.run(self.score_completion_async(*args, **kwargs))

# --- The rest of your utilities keep as-is (common_prefix_len_ids, compute_ref_diff, KL/JS helpers) ---

def common_prefix_len_ids(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def compute_ref_diff(
    tokenizer,
    raw_text: str,
    corr_text: str,
    raw_score: dict[str, Any],
    corr_score: dict[str, Any],
) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    diff["raw_sum_logp"] = raw_score["sum_logp"]
    diff["raw_avg_logp"] = raw_score["avg_logp"]
    diff["raw_n_tokens"] = raw_score["n_tokens"]
    diff["corr_sum_logp"] = corr_score["sum_logp"]
    diff["corr_avg_logp"] = corr_score["avg_logp"]
    diff["corr_n_tokens"] = corr_score["n_tokens"]
    diff["delta_sum_logp"] = corr_score["sum_logp"] - raw_score["sum_logp"]
    diff["delta_avg_logp"] = corr_score["avg_logp"] - raw_score["avg_logp"]

    raw_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    corr_ids = tokenizer.encode(corr_text, add_special_tokens=False)
    L = common_prefix_len_ids(raw_ids, corr_ids)
    diff["common_prefix_len"] = L
    if L > 0:
        raw_lp = raw_score["token_logprobs"][:L]
        corr_lp = corr_score["token_logprobs"][:L]
        d = [c - r for c, r in zip(corr_lp, raw_lp)]
        diff["sum_dlogp_prefix"] = float(sum(d))
        diff["avg_dlogp_prefix"] = float(sum(d) / len(d))
    return diff


def _softmax_from_logps(logps: list[float], temperature: float = 1.0) -> list[float]:
    xs = []
    for lp in logps:
        if lp == float("-inf") or math.isnan(lp):
            xs.append(float("-inf"))
        else:
            xs.append(lp / max(1e-8, temperature))

    m = max([x for x in xs if x != float("-inf")], default=float("-inf"))
    if m == float("-inf"):
        n = len(logps)
        return [1.0 / n] * n if n > 0 else []

    exps = [0.0 if x == float("-inf") else math.exp(x - m) for x in xs]
    s = sum(exps)
    if s <= 0:
        n = len(logps)
        return [1.0 / n] * n if n > 0 else []
    return [e / s for e in exps]


def kl_divergence(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        pi = max(eps, float(pi))
        qi = max(eps, float(qi))
        s += pi * math.log(pi / qi)
    return float(s)


def js_divergence(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def compute_tokenpos_kl_js(
    raw_score: dict[str, Any],
    corr_score: dict[str, Any],
    *,
    prefix_len: int,
    temperature: float = 1.0,
) -> dict[str, Any]:
    raw_lps = (raw_score.get("token_logprobs") or [])[:prefix_len]
    corr_lps = (corr_score.get("token_logprobs") or [])[:prefix_len]
    if prefix_len <= 0 or len(raw_lps) != prefix_len or len(corr_lps) != prefix_len:
        return {"prefix_len": prefix_len, "kl": None, "js": None}

    p = _softmax_from_logps(raw_lps, temperature=temperature)
    q = _softmax_from_logps(corr_lps, temperature=temperature)
    kl = kl_divergence(p, q)
    js = js_divergence(p, q, eps=1e-12)
    return {"prefix_len": prefix_len, "kl": kl, "js": js}
