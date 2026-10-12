"""
Rollout adapter: runs claw-eval's agent loop logic with SGLang-backed inference.

Each agent turn produces one Sample (for step-level credit assignment in RL).
Tool dispatch reuses claw-eval's ToolDispatcher / SandboxToolDispatcher.
"""

import asyncio
import base64
import io
import itertools
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from argparse import Namespace
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from qqr.data.prompts.qwen3 import Qwen3Prompt
from qqr.rollout.agent_rollout import GenerateState
from qqr.rollout.agent_rollout import generate as base_generate
from qqr.schemas import Sample

# RTI multi-stage inspector: reuse the shared pipeline (judge -> corrector ->
# postcheck -> ref_gate) + step calibration, exactly like qqr.examples.tau2_bench
# instead of a bespoke single-stage self-correct.
from qqr.off_policy.rti.inspector import RTIInspector
from qqr.off_policy.step_calibration import calibrate_step

from . import config
from .reward_model import eval_reward

logger = logging.getLogger(__name__)

# ── Output file detection for artifact collection ─────────────
_OUTPUT_PATH_RE = re.compile(
    r"/workspace/output\.\w+"
    r"|/workspace/[a-zA-Z0-9_-]+\.(?:html|json|csv|txt|py|js|css|svg|md)"
)
_TEXT_EXTS = {".html", ".json", ".csv", ".txt", ".py", ".js", ".css", ".svg", ".md", ".xml"}

# ── Tool call parser for Qwen3.5 <function=...> format ───────────
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function\s*=\s*([a-zA-Z0-9_:-]+)\s*>(.*?)</function>", re.IGNORECASE | re.DOTALL)
_PARAM_RE = re.compile(r"<parameter\s*=\s*([a-zA-Z0-9_:-]+)\s*>(.*?)</parameter>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _coerce_param(raw: str):
    """Convert parameter value string to appropriate Python type."""
    v = raw.strip()
    if not v:
        return ""
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def parse_qwen35_response(text: str) -> dict:
    """Parse Qwen3.5 response supporting both JSON and <function=...> tool call formats.

    On parse failure the original tool_call block is kept in content and no tool
    call is dispatched, so the model sees its broken output and the trajectory
    terminates instead of continuing with an empty-args call.
    """
    message = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}

    if "<think>" in text:
        m = _THINK_RE.search(text)
        if m:
            message["reasoning_content"] = m.group(1).strip()
            text = _THINK_RE.sub("", text)
    elif "</think>" in text:
        # Handle case where the open tag was part of the generation prompt.
        # The response starts directly with reasoning content followed by the close tag.
        think_end_idx = text.index("</think>")
        message["reasoning_content"] = text[:think_end_idx].strip()
        text = text[think_end_idx + len("</think>"):].strip()

    tc_matches = list(_TOOL_CALL_RE.finditer(text))
    successfully_parsed = []
    for idx, match in enumerate(tc_matches):
        block = match.group(1).strip()
        parsed_tc = None

        # Try JSON format first (Qwen3 style)
        try:
            data = json.loads(block)
            func_name = data.get("name")
            if not func_name:
                logger.warning(f"[claw-rollout] tool_call block missing 'name': {block[:80]}...")
                continue
            func_args = data.get("arguments", {})
            if isinstance(func_args, (dict, list)):
                func_args_str = json.dumps(func_args, ensure_ascii=False)
            else:
                func_args_str = str(func_args)
            parsed_tc = {
                "id": f"call_{idx + 1}",
                "type": "function",
                "function": {"name": func_name, "arguments": func_args_str},
            }
        except (json.JSONDecodeError, ValueError):
            pass

        # Try <function=name><parameter=key>value</parameter></function> format (Qwen3.5)
        if parsed_tc is None:
            func_match = _FUNCTION_RE.search(block)
            if func_match:
                func_name = func_match.group(1)
                param_block = func_match.group(2)
                params = {}
                for pm in _PARAM_RE.finditer(param_block):
                    params[pm.group(1)] = _coerce_param(pm.group(2))

                # If param_block has content but zero parameters were parsed,
                # this is a format error -- do not create an empty-args tool call.
                if param_block.strip() and not params:
                    logger.warning(
                        f"[claw-rollout] Tool call format error: {func_name} has content "
                        f"in param_block but no parameters parsed. Raw: {param_block[:100]}"
                    )
                    continue

                parsed_tc = {
                    "id": f"call_{idx + 1}",
                    "type": "function",
                    "function": {"name": func_name, "arguments": json.dumps(params, ensure_ascii=False)},
                }

        if parsed_tc is None:
            logger.warning(f"[claw-rollout] Could not parse tool_call block: {block[:80]}...")
            continue

        message["tool_calls"].append(parsed_tc)
        successfully_parsed.append(match)

    # Only remove successfully parsed tool_call blocks from content.
    # Failed parses remain so the model can see its broken output.
    remaining = text
    for match in reversed(successfully_parsed):
        remaining = remaining[:match.start()] + remaining[match.end():]
    eos = "<|im_end|>"
    message["content"] = remaining.removesuffix(eos).strip()
    return message

# ── Lazy import claw-eval (add to sys.path once, via qqr.tools.claw_eval bridge) ─
from qqr.tools.claw_eval import ensure_claw_eval_on_path as _ensure_claw_eval_on_path

_CLAW_SRC = _ensure_claw_eval_on_path()


# ── Port slot counter for unique service ports ─────────────────
_port_slot_counter = itertools.count()


def _is_port_free(port: int) -> bool:
    """Check if a port is available for binding."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            return True
    except OSError:
        return False


# ── Shared service manager (reference-counted) ─────────────────
_svc_lock = threading.Lock()
_svc_managers: dict[str, Any] = {}  # key → (ServiceManager, ref_count)


def _resolve_service_env(services, cwd) -> None:
    """Rewrite relative service env paths against the claw-eval data roots.

    ``ServiceManager._spawn`` only absolutizes keys ending in ``_FIXTURES``, but
    task.yaml files also carry relative paths under other keys (e.g.
    ``DOCUMENTS_BASE_DIR: fixtures/<task_id>``) which the service then resolves
    against its own cwd.  The vendored mock_services/ tree has no fixtures/
    sibling, so those values are rewritten here to absolute paths.
    """
    roots = [Path(cwd), config.fixtures_dir.parent]
    roots += [d.parent for d in config.tasks_dirs]
    for svc in services:
        env = getattr(svc, "env", None)
        if not env:
            continue
        for key, value in list(env.items()):
            if not isinstance(value, str) or not value or key == "PORT":
                continue
            rel = Path(value)
            if rel.is_absolute() or (Path(cwd) / rel).exists():
                continue
            for root in roots:
                candidate = root / rel
                if candidate.exists():
                    env[key] = str(candidate)
                    break


@contextmanager
def _shared_service_context(key: str, services, cwd, mock_today):
    """Reference-counted ServiceManager: first user starts, last user stops."""
    _needs_init = False
    with _svc_lock:
        if key in _svc_managers:
            val = _svc_managers[key]
            if val[0] is None:
                # Another thread is initializing this key — wait
                raise RuntimeError(f"service key {key} is being initialized by another thread")
            mgr, count = val
            _svc_managers[key] = (mgr, count + 1)
        else:
            _svc_managers[key] = (None, 0)  # placeholder: initializing
            _needs_init = True

    if _needs_init:
        from claw_eval.runner.services import ServiceManager
        _resolve_service_env(services, cwd)
        mgr = ServiceManager(services, cwd=cwd, mock_today=mock_today)
        try:
            mgr.__enter__()
        except Exception:
            with _svc_lock:
                del _svc_managers[key]
            raise
        with _svc_lock:
            _svc_managers[key] = (mgr, 1)

    try:
        yield mgr
    finally:
        _should_stop = False
        with _svc_lock:
            if key in _svc_managers:
                mgr, count = _svc_managers[key]
                if count <= 1:
                    del _svc_managers[key]
                    _should_stop = True
                else:
                    _svc_managers[key] = (mgr, count - 1)
        if _should_stop:
            mgr.__exit__(None, None, None)


def _needs_sandbox(task) -> bool:
    """判断是否需要启用E2B沙箱环境。
    
    启用条件（满足任一即可）：
    1. 有 sandbox_files 且无 services（传统T类任务）
    2. task.tools 中包含沙箱工具名称（如Bash）
    """
    from claw_eval.runner.sandbox_tools import SANDBOX_TOOL_NAMES
    
    # 条件1：传统的基于文件的沙箱需求
    has_sandbox_files = bool(getattr(task, "sandbox_files", None)) and not task.services
    
    # 条件2：task.tools中包含需要沙箱执行的工具
    tool_names = {t.name for t in getattr(task, "tools", [])}
    has_sandbox_tools = bool(tool_names & SANDBOX_TOOL_NAMES)
    
    return has_sandbox_files or has_sandbox_tools


_sandbox_semaphore: asyncio.Semaphore | None = None


def _get_sandbox_semaphore() -> asyncio.Semaphore:
    global _sandbox_semaphore
    if _sandbox_semaphore is None:
        sandbox_cfg = config._get_sandbox_config()
        max_concurrent = getattr(sandbox_cfg, "max_concurrent", 10) or 10
        _sandbox_semaphore = asyncio.Semaphore(max_concurrent)
    return _sandbox_semaphore


def _resolve_task_dir(task_id: str, sandbox_files: list[str]) -> str:
    task_dir = config.resolve_task_dir(task_id)
    if task_dir is None:
        task_dir = config.tasks_dir / task_id
    if sandbox_files and not (task_dir / sandbox_files[0]).exists():
        shared_dir = config.fixtures_dir / task_id
        if (shared_dir / sandbox_files[0]).exists():
            return str(shared_dir)
    return str(task_dir)


@asynccontextmanager
async def _sandbox_context(task, task_id: str):
    from claw_eval.runner.sandbox_runner import E2BSandboxRunner

    sandbox_cfg = config._get_sandbox_config()
    sem = _get_sandbox_semaphore()
    loop = asyncio.get_running_loop()
    handle = None
    sandbox = None
    await sem.acquire()
    try:
        runner = E2BSandboxRunner(sandbox_cfg)
        run_id = f"rl-{task_id}-{uuid.uuid4().hex[:8]}"
        # Guard against SDK-level hangs (e.g. e2b control-plane HTTP stuck in
        # ssl.read with no timeout): Sandbox.create waits for the sandbox to
        # become ready (metadata wait_ready_timeout), so allow that budget
        # plus slack before giving up. On timeout the group falls back to the
        # "sandbox unavailable" padding path instead of freezing the rollout.
        create_timeout = max(360.0, float(sandbox_cfg.wait_ready_timeout or 300) + 60.0)
        try:
            handle = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: runner.start_sandbox(run_id=run_id)),
                timeout=create_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[claw-rollout] start_sandbox timed out after {create_timeout:.0f}s "
                f"for {run_id} — treating sandbox as unavailable"
            )
            handle = None
        if handle is not None:
            task_dir = _resolve_task_dir(task_id, task.sandbox_files or [])
            n_injected = await loop.run_in_executor(None, lambda: runner.inject_files(handle, task, task_dir=task_dir))
            expected = len(task.sandbox_files) if task.sandbox_files else 0
            if expected and n_injected < expected:
                logger.warning(f"[claw-rollout] Sandbox {task_id}: only {n_injected}/{expected} files injected (task_dir={task_dir})")
            sandbox = handle.sandbox
    except Exception as e:
        logger.warning(f"[claw-rollout] Sandbox failed for {task_id}: {e}")
    try:
        yield sandbox
    finally:
        if handle is not None:
            try:
                await loop.run_in_executor(None, lambda: runner.stop_sandbox(handle))
            except Exception as e:
                logger.warning(f"[claw-rollout] Sandbox cleanup failed: {e}")
        sem.release()


@asynccontextmanager
async def _service_context_async(svc_key, services, cwd, mock_today):
    loop = asyncio.get_running_loop()
    ctx = _shared_service_context(svc_key, services, cwd=cwd, mock_today=mock_today)
    await loop.run_in_executor(None, ctx.__enter__)
    try:
        yield None
    finally:
        try:
            await asyncio.shield(
                loop.run_in_executor(None, lambda: ctx.__exit__(None, None, None))
            )
        except asyncio.CancelledError:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            raise


_user_agent_cache: dict[str, Any] = {}


def _get_user_agent(task):
    ua_cfg = task.user_agent
    if not ua_cfg.enabled:
        return None
    model_key = "default"
    if model_key in _user_agent_cache:
        return _user_agent_cache[model_key]
    from claw_eval.runner.user_agent import UserAgent
    ua_model_cfg = config._get_user_agent_config()
    api_key = ua_model_cfg.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    agent = UserAgent(
        model_id=ua_model_cfg.model_id,
        api_key=api_key,
        base_url=ua_model_cfg.base_url,
    )
    extra_body = getattr(ua_model_cfg, "extra_body", None)
    if extra_body:
        _orig_create = agent.client.chat.completions.create
        def _create_with_extra_body(**kwargs):
            kwargs.setdefault("extra_body", extra_body)
            return _orig_create(**kwargs)
        agent.client.chat.completions.create = _create_with_extra_body
    _user_agent_cache[model_key] = agent
    return agent


def _load_claw_task(task_id: str):
    """Load a claw-eval TaskDefinition by task_id."""
    from claw_eval.models.task import TaskDefinition

    task_dir = config.resolve_task_dir(task_id)
    if task_dir is None:
        raise FileNotFoundError(f"Task not found: {task_id} (looked in {config.tasks_dirs})")
    return TaskDefinition.from_yaml(task_dir / "task.yaml")


def _build_claw_system_prompt(task, extra_tools=None) -> str:
    """Build claw-eval system prompt for a task."""
    from claw_eval.config import PromptConfig
    from claw_eval.runner.system_prompt import build_system_prompt

    prompt_cfg = PromptConfig(strict_file_check=False)
    return build_system_prompt(task, prompt_cfg=prompt_cfg, extra_tools=extra_tools)


def _build_tool_defs_for_chat_template(task, sandbox_tool_list=None) -> list[dict]:
    """Convert claw-eval ToolSpec list → OpenAI function-calling dicts for chat template."""
    all_tools = list(task.tools) + (sandbox_tool_list or [])
    result = []
    for spec in all_tools:
        result.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        })
    return result


def _make_dispatcher(task, sandbox_tools: bool, sandbox=None):
    """Create the appropriate tool dispatcher for a task."""
    from claw_eval.runner.dispatcher import ToolDispatcher

    endpoint_map = task.get_endpoint_map()
    http_dispatcher = ToolDispatcher(endpoint_map)

    if not sandbox_tools:
        return http_dispatcher, None

    from claw_eval.runner.sandbox_dispatcher import SandboxToolDispatcher
    from claw_eval.runner.sandbox_tools import SANDBOX_TOOLS

    existing_names = {t.name for t in task.tools}
    sandbox_tool_list = [t for t in SANDBOX_TOOLS if t.name not in existing_names]
    dispatcher = SandboxToolDispatcher(http_dispatcher, sandbox=sandbox)
    return dispatcher, sandbox_tool_list


def _image_blocks_to_pil(image_blocks) -> list:
    """Convert claw-eval ImageBlocks (base64 JPEG/PNG) to PIL Images."""
    images = []
    for block in image_blocks:
        raw = base64.b64decode(block.data)
        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
        images.append(img)
    return images


def _count_image_blocks(messages: list[dict]) -> int:
    """Count the total number of image content blocks in messages."""
    count = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                count += 1
    return count


def _extract_initial_images(messages: list[dict], sample_multimodal_inputs: dict | None) -> list:
    """Extract initial images from sample's multimodal_inputs to match image blocks in messages.

    When the initial query/prompt contains image content blocks, the corresponding
    PIL images must be included in accumulated_images so that apply_chat_template
    output (with image placeholders) is consistent with the actual image count.
    """
    n_blocks = _count_image_blocks(messages)
    if n_blocks == 0:
        return []

    # Try to get images from sample.multimodal_inputs
    if sample_multimodal_inputs and sample_multimodal_inputs.get("images"):
        images = list(sample_multimodal_inputs["images"])
        if len(images) >= n_blocks:
            return images[:n_blocks]
        # Partial match - return what we have
        return images

    # Try to extract PIL images directly from message content blocks (base64/path)
    images = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image_data = item.get("image")
            if image_data is None:
                continue
            if isinstance(image_data, PILImage.Image):
                images.append(image_data)
            elif isinstance(image_data, str) and not image_data.startswith("placeholder:"):
                try:
                    if image_data.startswith("data:"):
                        _, encoded = image_data.split(",", 1)
                        raw = base64.b64decode(encoded)
                    else:
                        raw = base64.b64decode(image_data)
                    images.append(PILImage.open(io.BytesIO(raw)).convert("RGB"))
                except Exception:
                    pass
    return images


def _sync_images_and_messages(messages: list[dict], accumulated_images: list) -> None:
    """Ensure image blocks in messages are in sync with accumulated_images.

    If there are more image blocks in messages than images in accumulated_images,
    remove the excess image blocks (from the beginning) and replace with text
    placeholders. This prevents IndexError in the processor.
    """
    n_blocks = _count_image_blocks(messages)
    n_images = len(accumulated_images)

    if n_blocks <= n_images:
        return

    n_drop = n_blocks - n_images
    logger.warning(
        f"[claw-rollout] Image sync: {n_blocks} image blocks but {n_images} images, "
        f"removing {n_drop} excess block(s) from messages"
    )

    dropped = 0
    for msg in messages:
        if dropped >= n_drop:
            break
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image" and dropped < n_drop:
                new_content.append({"type": "text", "text": "[image removed]"})
                dropped += 1
            else:
                new_content.append(item)
        msg["content"] = new_content


def _cap_images(messages: list[dict], accumulated_images: list, max_images: int) -> None:
    """Drop image content blocks from oldest messages when total exceeds max_images.

    Modifies messages and accumulated_images in-place.  Keeps the most recent
    images; replaces earlier image blocks with text placeholders so that
    len(accumulated_images) always equals the number of {"type": "image"} blocks
    remaining in messages.
    """
    if max_images <= 0 or len(accumulated_images) <= max_images:
        return

    n_drop = len(accumulated_images) - max_images

    dropped = 0
    for msg in messages:
        if dropped >= n_drop:
            break
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image" and dropped < n_drop:
                # Replace image block with text placeholder to keep messages
                # consistent: no orphan <|image|> tokens will be generated.
                new_content.append({"type": "text", "text": "[image removed]"})
                dropped += 1
            else:
                new_content.append(item)
        msg["content"] = new_content

    del accumulated_images[:n_drop]


def _dispatch_tool_calls(dispatcher, tool_calls: list[dict], trace_id: str = "rl", task=None) -> tuple[list[dict], list]:
    """Execute tool calls via claw-eval dispatcher and return (tool_messages, pil_images)."""
    from claw_eval.models.content import TextBlock, ToolUseBlock

    tool_messages = []
    all_new_images = []
    for tc in tool_calls:
        func = tc["function"]
        tool_name = func["name"]
        if not tool_name:
            logger.warning(f"[claw-rollout] skipping tool_call with missing name: {tc}")
            continue
        try:
            args = json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"]
        except json.JSONDecodeError:
            args = {}

        tool_use = ToolUseBlock(
            id=tc.get("id", "call_0"),
            name=tool_name,
            input=args,
        )

        try:
            result = dispatcher.dispatch(tool_use, trace_id)
            extra_images = None
            if len(result) == 3:
                tool_result, _, extra_images = result
            else:
                tool_result, _ = result

            content_text = ""
            for block in tool_result.content:
                if hasattr(block, "text"):
                    content_text += block.text

            if len(content_text) > config.tool_result_max_chars:
                original_len = len(content_text)
                file_hint = ""
                if task and getattr(task, "sandbox_files", None):
                    ws_files = [
                        f"/workspace/{f}" for f in task.sandbox_files
                        if Path(f).suffix.lower() in {".txt", ".csv", ".json", ".md", ".log"}
                    ]
                    if ws_files:
                        file_hint = f" Full extracted text is available at: {', '.join(ws_files)}."
                content_text = (
                    content_text[: config.tool_result_max_chars]
                    + f"\n\n[truncated: showing {config.tool_result_max_chars:,} of {original_len:,} characters."
                    + file_hint
                    + " Use Read/Grep/Bash to inspect the full content.]"
                )

            if extra_images:
                pil_images = _image_blocks_to_pil(extra_images)
                all_new_images.extend(pil_images)

            is_error = tool_result.is_error
        except Exception as e:
            content_text = f"[Error] Tool dispatch failed: {e}"
            is_error = True

        tool_messages.append({
            "role": "tool",
            "content": content_text,
            "tool_call_id": tc.get("id", "call_0"),
        })

    return tool_messages, all_new_images


# ── Output artifact collection ─────────────────────────────────────

def _detect_output_paths(task) -> list[str]:
    """Extract output file paths mentioned in the task prompt."""
    prompt_text = task.prompt.text if hasattr(task.prompt, "text") else str(task.prompt)
    paths = _OUTPUT_PATH_RE.findall(prompt_text)
    for pattern in getattr(task, "env_snapshot_files", []):
        if "*" not in pattern and "?" not in pattern:
            ext = Path(pattern).suffix.lower()
            if ext in _TEXT_EXTS and pattern not in paths:
                paths.append(pattern)
    return list(dict.fromkeys(paths))


async def _collect_output_artifacts(sandbox, task) -> str | None:
    """Read text output files from sandbox after agent loop completes."""
    if sandbox is None:
        return None
    output_paths = _detect_output_paths(task)
    if not output_paths:
        return None
    artifacts = []
    loop = asyncio.get_running_loop()
    for path in output_paths:
        try:
            content = await loop.run_in_executor(
                None, lambda p=path: sandbox.files.read(p)
            )
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if content.strip():
                artifacts.append(f"[File: {path}]\n{content}")
        except Exception:
            pass
    if not artifacts:
        return None
    combined = "\n\n---\n\n".join(artifacts)
    return combined[:config.artifact_max_chars]


# ── Env snapshot collection (rule-based grader verification) ────

def _has_env_snapshot_commands(task) -> bool:
    """Check if a task defines env_snapshot_commands (for rule-based graders)."""
    return bool(getattr(task, "env_snapshot_commands", None))


async def _collect_env_snapshot(sandbox, task, task_id: str) -> dict | None:
    """Run env_snapshot_commands in the sandbox and collect their stdout/stderr.

    This produces the env_snapshot dict that rule-based graders (e.g.
    XssFilterGrader, WalRecoveryGrader) expect:

        {
            "cmd:python /workspace/fixtures/verify_filter.py": {"stdout": "...", "stderr": ""},
            "cmd:test -f /workspace/filter.py && echo EXISTS || echo MISSING": {"stdout": "EXISTS\\n", "stderr": ""},
        }

    Must be called *before* the sandbox is torn down.
    """
    if sandbox is None:
        return None
    if not _has_env_snapshot_commands(task):
        return None

    loop = asyncio.get_running_loop()
    env_timeout = 30
    if hasattr(task, "environment") and hasattr(task.environment, "env_snapshot_timeout"):
        env_timeout = task.environment.env_snapshot_timeout or 30

    # Step 1: Inject grader files (e.g. verify_*.py) into the sandbox
    await loop.run_in_executor(None, lambda: _inject_grader_files_sync(sandbox, task, task_id))

    # Step 2: Run each env_snapshot_command and collect stdout/stderr
    snapshot: dict[str, dict[str, str]] = {}
    for cmd in getattr(task, "env_snapshot_commands", []):
        if not cmd:
            continue
        try:
            result = await loop.run_in_executor(
                None, lambda c=cmd: sandbox.commands.run(c, timeout=env_timeout)
            )
            snapshot[f"cmd:{cmd}"] = {
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            }
            if result.exit_code != 0:
                logger.warning(
                    f"[claw-rollout] env_snapshot cmd failed (exit={result.exit_code}): {cmd[:80]}"
                )
        except Exception as e:
            logger.warning(f"[claw-rollout] env_snapshot cmd error: {e}")
            snapshot[f"cmd:{cmd}"] = {"stdout": "", "stderr": str(e)}

    return snapshot if snapshot else None


# ── Screenshot collection for visual judge ────────────────────────

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _is_image_pattern(pattern: str) -> bool:
    """Check if a file pattern (possibly a glob) refers to image files."""
    if "*" not in pattern and "?" not in pattern:
        return Path(pattern).suffix.lower() in _IMAGE_EXTS
    # For globs: check if any image extension appears in the pattern
    low = pattern.lower()
    return any(ext in low for ext in _IMAGE_EXTS)


def _has_snapshot_images(task) -> bool:
    """Check if a task has env_snapshot infrastructure that produces images."""
    return any(_is_image_pattern(p) for p in getattr(task, "env_snapshot_files", []))


def _inject_grader_files_sync(sandbox, task, task_id: str) -> int:
    """Inject sandbox_grader_files into the sandbox for post-loop snapshot commands."""
    grader_files = getattr(task, "sandbox_grader_files", None)
    if not grader_files:
        return 0

    task_dir = config.resolve_task_dir(task_id)
    if task_dir is None:
        task_dir = config.tasks_dir / task_id
    injected = 0
    for rel_path in grader_files:
        src = task_dir / rel_path
        if not src.exists():
            src = config.fixtures_dir / task_id / rel_path
        if not src.exists():
            logger.warning(f"[claw-rollout] grader file not found: {rel_path}")
            continue

        container_path = f"/workspace/{rel_path}"
        parent_dir = str(Path(container_path).parent)
        try:
            sandbox.commands.run(f"mkdir -p {parent_dir}", timeout=5)
        except Exception:
            pass

        try:
            content = src.read_bytes()
            sandbox.files.write(container_path, content)
            injected += 1
        except Exception as e:
            logger.warning(f"[claw-rollout] grader inject failed {rel_path}: {e}")

    return injected


def _read_image_b64_sync(sandbox, file_path: str) -> dict | None:
    """Read an image file from sandbox as base64."""
    try:
        result = sandbox.commands.run(f"base64 '{file_path}'", timeout=30)
        if result.exit_code != 0:
            return None
        b64 = result.stdout.strip().replace("\n", "")
        if not b64:
            return None
        import mimetypes
        mime, _ = mimetypes.guess_type(file_path)
        return {
            "path": file_path,
            "base64": b64,
            "mime_type": mime or "image/png",
        }
    except Exception:
        return None


def _resolve_snapshot_file_paths_sync(sandbox, patterns: list[str]) -> list[str]:
    """Resolve env_snapshot_files patterns (including globs) to concrete paths."""
    paths = []
    for pattern in patterns:
        if not _is_image_pattern(pattern):
            continue

        if "*" in pattern or "?" in pattern:
            try:
                result = sandbox.commands.run(
                    f"python3 -c \"import glob, json; print(json.dumps(sorted(glob.glob('{pattern}'))))\"",
                    timeout=10,
                )
                import json as _json
                file_list = _json.loads(result.stdout.strip())
                paths.extend(file_list)
            except Exception:
                pass
        else:
            paths.append(pattern)
    return paths


async def _collect_screenshots(sandbox, task, task_id: str) -> list[dict] | None:
    """Collect screenshot images from sandbox for visual judge evaluation."""
    if sandbox is None:
        return None
    if not _has_snapshot_images(task):
        return None

    loop = asyncio.get_running_loop()
    env_timeout = 30
    if hasattr(task, "environment") and hasattr(task.environment, "env_snapshot_timeout"):
        env_timeout = task.environment.env_snapshot_timeout or 30

    # Step 1: Inject grader files (e.g. take_recording.py)
    await loop.run_in_executor(None, lambda: _inject_grader_files_sync(sandbox, task, task_id))

    # Step 2: Run env_snapshot_commands (generate screenshots)
    for cmd in getattr(task, "env_snapshot_commands", []):
        if not cmd:
            continue
        try:
            result = await loop.run_in_executor(
                None, lambda c=cmd: sandbox.commands.run(c, timeout=env_timeout)
            )
            if result.exit_code != 0:
                logger.warning(f"[claw-rollout] snapshot cmd failed (exit={result.exit_code}): {cmd[:80]}")
        except Exception as e:
            logger.warning(f"[claw-rollout] snapshot cmd error: {e}")

    # Step 3: Resolve and read image files
    snapshot_files = getattr(task, "env_snapshot_files", [])
    image_paths = await loop.run_in_executor(
        None, lambda: _resolve_snapshot_file_paths_sync(sandbox, snapshot_files)
    )
    if not image_paths:
        return None

    # Step 4: Read images as base64, limiting count
    max_images = config.screenshot_max_images
    if len(image_paths) > max_images:
        step = len(image_paths) / max_images
        image_paths = [image_paths[int(i * step)] for i in range(max_images)]

    screenshots = []
    for path in image_paths:
        img = await loop.run_in_executor(None, lambda p=path: _read_image_b64_sync(sandbox, p))
        if img:
            screenshots.append(img)

    return screenshots if screenshots else None


# ── Main rollout entry point ─────────────────────────────────────


def _build_inspector(args: Namespace):
    """Build the shared multi-stage RTI inspector (tau2-bench parity).

    Mirrors qqr.examples.tau2_bench.rollout._build_inspector: constructs the
    judge -> corrector -> postcheck -> ref_gate CalibrationPipeline via
    RTIInspector. The judge/corrector/postcheck LLM calls default to the
    running SGLang policy endpoint (OpenAI-compat) unless inspector-openai-* is
    set; prompts fall back to qqr.off_policy.rti.prompts defaults unless
    custom-inspector-prompt-path is configured.

    If ``custom-inspector-method-path`` is set (e.g. qqr.off_policy.luffy.inspector),
    dynamically load and delegate to that module's ``build_inspector(cfg, args)``
    instead, allowing claw-eval to use Luffy or other off-policy calibrators.

    ``enabled`` is forced True because this is only called from Stage-2
    self-correction (inspector_enabled=True); per-sample gating is handled by
    calibrate_step via sample.label['inspector'] (set by group_rollout).
    """
    from qqr.examples.tau2_bench.config import InspectorConfig

    use_judge_llm = bool(getattr(args, "inspector_use_judge_llm", False))
    openai_base_url = getattr(args, "inspector_openai_base_url", None)
    openai_api_key = getattr(args, "inspector_openai_api_key", "EMPTY")
    openai_model = getattr(args, "inspector_openai_model", None)

    if openai_base_url is None or openai_model is None:
        openai_base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}".rstrip("/") + "/v1"
        openai_model = (
            getattr(args, "ref_model_id", None)
            or getattr(args, "model", None)
            or getattr(args, "hf_checkpoint", None)
        )
        openai_api_key = "EMPTY"

    cfg = InspectorConfig(
        enabled=True,
        use_judge_llm=use_judge_llm,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        enable_postcheck=bool(getattr(args, "enable_postcheck", False)),
        temperature=float(getattr(args, "inspector_temperature", 0.0)),
        max_new_tokens=int(getattr(args, "inspector_max_new_tokens", 800)),
        timeout=int(getattr(args, "inspector_timeout", 300)),
        allow_gt_bootstrap=bool(getattr(args, "inspector_allow_gt_bootstrap", False)),
        strict_direction_match=bool(getattr(args, "inspector_strict_direction_match", False)),
    )

    # Dynamic dispatch to a custom off-policy inspector (e.g. Luffy).
    custom_inspector_method_path = getattr(args, "custom_inspector_method_path", None)
    if custom_inspector_method_path:
        import importlib
        mod = importlib.import_module(custom_inspector_method_path)
        return mod.build_inspector(cfg, args)

    return RTIInspector(cfg, args=args)


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    inspector_enabled: bool = False,
) -> Sample | list[Sample]:
    """
    Custom generate function for slime: runs claw-eval agent loop with SGLang.

    Args:
        args: slime training args (contains sglang config, model paths, etc.)
        sample: Sample with prompt (task query) and metadata (task_id)
        sampling_params: SGLang sampling parameters
        evaluation: if True, compute eval reward and return single Sample

    Returns:
        list[Sample] for training (one per agent step), or single Sample for eval.
    """
    # Propagate train_env_vars to this process (Ray workers don't inherit parent env)
    env_vars = getattr(args, "train_env_vars", None)
    if env_vars:
        if isinstance(env_vars, str):
            try:
                env_vars = json.loads(env_vars)
            except Exception as e:
                logger.warning(f"[claw-rollout] Failed to parse train_env_vars JSON: {e}")
                env_vars = None
        if isinstance(env_vars, dict):
            for k, v in env_vars.items():
                os.environ[k] = str(v)

    task_id = sample.metadata.get("task_id") if sample.metadata else None
    if not task_id:
        raise ValueError("sample.metadata must contain 'task_id'")

    task = _load_claw_task(task_id)

    # Normalize sample.prompt in place: if a user message's content is a list of
    # single-char text items (char-split format), merge them back into a single
    # string. This must run BEFORE query extraction so all downstream paths
    # (Path B L803, Path C L872, Path D L925) reuse the normalized prompt.
    # Note: only merge content lists that are ALL text items; lists containing
    # {'type': 'image', ...} are valid multimodal content and must be left intact.
    if isinstance(sample.prompt, list):
        for msg in sample.prompt:
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                content = msg["content"]
                if content and all(
                    isinstance(item, dict) and item.get("type") == "text"
                    for item in content
                ):
                    msg["content"] = "".join(item.get("text", "") for item in content)

    if isinstance(sample.prompt, str):
        query = sample.prompt
    else:
        query = sample.prompt[-1]["content"]

    # Normalize query: if dataset provides content as a list of single-char text items,
    # merge them back into a single string. This can happen when the dataset serializes
    # characters individually in the multimodal content format.
    if isinstance(query, list):
        if all(isinstance(item, dict) and item.get("type") == "text" for item in query):
            query = "".join(item.get("text", "") for item in query)
        elif all(isinstance(item, str) for item in query):
            query = "".join(query)
    samples = await _agent_loop(args, sample, sampling_params, task, query, evaluation=evaluation, inspector_enabled=inspector_enabled)

    if evaluation:
        final_sample = samples[-1]
        await eval_reward(args, final_sample)
        return final_sample
    else:
        return samples


async def _agent_loop(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    task,
    query: str,
    evaluation: bool = False,
    inspector_enabled: bool = False,
) -> list[Sample]:
    """
    Core agent loop mirroring claw-eval's run_task(), but with SGLang inference.

    For each turn:
      1. Build messages with system prompt + conversation history
      2. Call SGLang via base_generate() to get model response + tokens/logprobs
      3. Parse tool calls from response
      4. Dispatch tools via claw-eval's dispatcher
      5. Collect Sample for step-level training
    """
    state = GenerateState(args)
    eos_token = state.tokenizer.eos_token or "<|im_end|>"
    max_steps = min(config.max_steps, task.environment.max_turns)

    # RTI Inspector initialization — shared multi-stage pipeline (tau2 parity)
    _inspector_instance = None
    if inspector_enabled:
        _inspector_instance = _build_inspector(args)

    # Port offset: each concurrent trajectory gets a unique port range to
    # prevent service bind collisions across prompts and trajectories.
    # Use optimistic allocation with retry on ServiceStartError.
    from claw_eval.runner.services import ServiceStartError

    worker_rank = int(os.getenv("SLIME_WORKER_RANK", "0"))
    port_cycle = 2048
    port_multiplier = 25  # Keep total range within 65535 (2048*25=51200 max offset)
    max_retries = 15

    samples = []
    task_id = sample.metadata["task_id"]
    needs_sandbox = _needs_sandbox(task)
    has_services = bool(task.services)

    # --- Phase 1: Start HTTP services if needed (independent of sandbox) ---
    svc_inner_ctx = None  # Track for cleanup in combined context
    if has_services:
        cumulative_offset = 0
        last_error = None
        for _retry in range(max_retries):
            slot = next(_port_slot_counter) % port_cycle + 1
            port_offset = (worker_rank * port_cycle + slot) * port_multiplier
            delta = port_offset - cumulative_offset
            task.apply_port_offset(delta)
            cumulative_offset = port_offset

            # Pre-check: verify candidate ports are free before attempting start
            candidate_ports = [svc.port for svc in task.services]
            port_busy = False
            for cp in candidate_ports:
                if cp > 65535 or cp < 1:
                    logger.warning(f"[claw-rollout] Port {cp} out of valid range (offset {port_offset}, slot {slot}), skipping")
                    port_busy = True
                    break
                if not _is_port_free(cp):
                    logger.warning(f"[claw-rollout] Port {cp} already in use (offset {port_offset}, slot {slot}), skipping to next slot")
                    port_busy = True
                    break
            if port_busy:
                continue

            # Exponential backoff on retries
            if _retry > 0:
                await asyncio.sleep(min(0.1 * (2 ** (_retry - 1)), 2.0))

            svc_key = f"{task_id}_{port_offset}"
            try:
                loop = asyncio.get_running_loop()
                svc_inner_ctx = _shared_service_context(svc_key, task.services, cwd=config.service_cwd, mock_today=task.environment.mock_today)
                await loop.run_in_executor(None, svc_inner_ctx.__enter__)
                break
            except ServiceStartError as e:
                svc_inner_ctx = None
                last_error = e
                logger.warning(f"[claw-rollout] Port conflict on slot {slot} (offset {port_offset}, ports {candidate_ports}), retry {_retry+1}/{max_retries}: {e}")
        else:
            raise last_error  # type: ignore[misc]

    # --- Phase 2: Build combined context (sandbox + service cleanup) ---
    if needs_sandbox:
        if not has_services:
            # Traditional T-class: sandbox only, apply port offset here
            slot = next(_port_slot_counter) % port_cycle + 1
            port_offset = (worker_rank * port_cycle + slot) * port_multiplier
            task.apply_port_offset(port_offset)

        @asynccontextmanager
        async def _combined_ctx():
            """Manage sandbox lifecycle with service cleanup guarantee."""
            try:
                async with _sandbox_context(task, task_id) as sandbox:
                    yield sandbox
            finally:
                if svc_inner_ctx is not None:
                    try:
                        svc_inner_ctx.__exit__(None, None, None)
                    except Exception:
                        pass

        ctx = _combined_ctx()
    else:
        if svc_inner_ctx is not None:
            # Services only (no sandbox): cleanup context
            @asynccontextmanager
            async def _svc_ctx_with_cleanup():
                try:
                    yield None
                finally:
                    try:
                        svc_inner_ctx.__exit__(None, None, None)
                    except Exception:
                        pass

            ctx = _svc_ctx_with_cleanup()
        else:
            @asynccontextmanager
            async def _noop_ctx():
                yield None
            ctx = _noop_ctx()

    async with ctx as sandbox:
        if needs_sandbox and sandbox is None:
            logger.warning(f"[claw-rollout] Sandbox unavailable for {task_id}, returning padding")
            # Single placeholder instead of (max_steps + 1) pads: fixed-length
            # padding is no longer needed (see the note at the end of this function).
            #
            # padding_num = max_steps + 1
            # for _ in range(padding_num):
            #     pad_sample = Sample(
            #         group_index=sample.group_index,
            #         index=sample.index,
            #         tokens=[state.tokenizer.pad_token_id],
            #         reward=0.0,
            #         loss_mask=[],
            #         rollout_log_probs=[],
            #     )
            #     pad_sample.group_id = sample.index
            #     samples.append(pad_sample)
            pad_sample = Sample(
                group_index=sample.group_index,
                index=sample.index,
                tokens=[state.tokenizer.pad_token_id],
                reward=0.0,
                loss_mask=[],
                rollout_log_probs=[],
                teacher_log_probs=[],
            )
            pad_sample.group_id = sample.index
            samples.append(pad_sample)
            return samples

        use_sandbox_tools = needs_sandbox and sandbox is not None
        dispatcher, sandbox_tool_list = _make_dispatcher(task, use_sandbox_tools, sandbox=sandbox)
        tool_defs = _build_tool_defs_for_chat_template(task, sandbox_tool_list)
        system_prompt = _build_claw_system_prompt(task, extra_tools=sandbox_tool_list)

        user_agent = _get_user_agent(task)
        ua_cfg = task.user_agent
        ua_enabled = ua_cfg.enabled and user_agent is not None
        ua_rounds = 0
        ua_max_rounds = ua_cfg.max_rounds if ua_enabled else 0

        if ua_enabled and ua_cfg.system_prompt_suffix:
            system_prompt = system_prompt + "\n\n" + ua_cfg.system_prompt_suffix

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Initialize accumulated_images with any images already present in the
        # initial messages (e.g. from a multimodal task prompt).  This ensures
        # that when apply_chat_template generates <|image|> placeholders for
        # these blocks, the corresponding PIL images are available in
        # multimodal_inputs.  When MULTIMODAL_INPUT_ENABLED is false, force a
        # text-only rollout by keeping the image buffer empty.
        accumulated_images = (
            _extract_initial_images(messages, sample.multimodal_inputs)
            if config.MULTIMODAL_INPUT_ENABLED
            else []
        )
        wall_start = time.monotonic()

        for step_idx in range(max_steps):
            elapsed = time.monotonic() - wall_start
            if elapsed > config.timeout_seconds:
                logger.warning(f"[claw-rollout] Timeout at step {step_idx}")
                break

            # Create Sample for this step
            # When images are accumulated, apply chat template to generate prompt with
            # vision token markers (<|vision_start|>...<|vision_end|>) so the processor
            # can insert image_pad tokens (ID 248056) into input_ids for training.
            if accumulated_images:
                step_prompt = state.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    tools=tool_defs if tool_defs else None,
                )
            else:
                step_prompt = sample.prompt

            cur_sample = Sample(
                group_index=sample.group_index,
                index=sample.index,
                messages=deepcopy(messages),
                prompt=step_prompt,
                label=sample.label,
                status=Sample.Status.PENDING,
                metadata=sample.metadata,
                train_metadata={"tools": tool_defs if tool_defs else None},
                multimodal_inputs={"images": list(accumulated_images)} if accumulated_images else {},
            )
            cur_sample.group_id = sample.index

            # Safety check: ensure image blocks in messages match accumulated_images
            _sync_images_and_messages(cur_sample.messages, accumulated_images)
            if cur_sample.multimodal_inputs:
                cur_sample.multimodal_inputs["images"] = list(accumulated_images)
            elif accumulated_images:
                cur_sample.multimodal_inputs = {"images": list(accumulated_images)}

            if step_idx == 1:
                print()

            # Call SGLang for model inference
            cur_sample = await base_generate(args, cur_sample, sampling_params)

            # RTI parity with tau2_bench (rollout.py:596): every training sample
            # carries teacher_log_probs by default (= behavior-policy logprobs,
            # deep-copied so later in-place writes don't alias). calibrate_step
            # overwrites it ONLY for corrected steps. Without this default,
            # a batch whose first flattened sample happens to be RTI-corrected
            # makes _convert_samples_to_train_data collect a mixed
            # [list, None, ...] teacher_log_probs list (it only checks
            # samples[0]), and the train side turns None into [] which trips
            # `assert len(log_prob) == response_length` in slice_log_prob_with_cp.
            cur_sample.teacher_log_probs = deepcopy(cur_sample.rollout_log_probs)

            # ─── RTI Per-Step Calibration (shared multi-stage pipeline) ───
            # calibrate_step runs RTIInspector's judge -> corrector -> postcheck
            # -> ref_gate pipeline, records the full inspector_trace into
            # cur_sample.metadata, and replaces response/tokens/log_probs in-place
            # when a correction passes the ref-KL gate. Per-sample gating is via
            # cur_sample.label['inspector'] (set by group_rollout Stage 2).
            if inspector_enabled and _inspector_instance is not None:
                _was_corrected, _info = await calibrate_step(
                    args, cur_sample, _inspector_instance, step_idx, sampling_params
                )
                if _was_corrected:
                    logger.info(f"[RTI] Step {step_idx} corrected for task {task_id}")
            # ─── End RTI ───

            # Parse the model response
            raw_response = cur_sample.response
            parsed = parse_qwen35_response(raw_response)
            cur_sample.response_message = parsed

            # Append assistant message to conversation
            assistant_msg = {
                "role": "assistant",
                "content": raw_response.removesuffix(eos_token),
                "raw": raw_response.removesuffix(eos_token),
            }
            messages.append(assistant_msg)

            tool_calls = parsed.get("tool_calls") or []

            if not tool_calls:
                samples.append(cur_sample)
                if ua_enabled and ua_rounds < ua_max_rounds:
                    from claw_eval.models.message import Message as CMessage

                    def _sanitize_content_for_cmessage(content):
                        """Convert message content to CMessage-compatible format.

                        Replaces image placeholder blocks with text descriptions
                        since user-agent only needs conversational context.
                        """
                        if isinstance(content, str):
                            return content
                        if not isinstance(content, list):
                            return content or ""
                        sanitized = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "image":
                                sanitized.append({"type": "text", "text": "[screenshot]"})
                            else:
                                sanitized.append(block)
                        return sanitized

                    claw_msgs = [CMessage(role=m["role"], content=_sanitize_content_for_cmessage(m.get("content", ""))) for m in messages if m["role"] != "tool"]
                    _loop = asyncio.get_running_loop()
                    ua_text = await _loop.run_in_executor(None, lambda: user_agent.generate_response(persona=ua_cfg.persona, conversation_messages=claw_msgs))
                    if ua_text is None:
                        break
                    ua_rounds += 1
                    logger.info(f"[claw-rollout] user-agent round {ua_rounds}/{ua_max_rounds}: {ua_text[:300]}")
                    messages.append({"role": "user", "content": f"[user_agent]\n{ua_text}"})
                    continue
                break

            # Dispatch tool calls via claw-eval (may block on E2B HTTP or local subprocess)
            _loop = asyncio.get_running_loop()
            tool_responses, new_images = await _loop.run_in_executor(None, lambda: _dispatch_tool_calls(dispatcher, tool_calls, task=task))
            messages.extend(tool_responses)

            # Inject images as a separate user message (matching claw-eval runner pattern)
            if new_images:
                if config.MULTIMODAL_INPUT_ENABLED:
                    accumulated_images.extend(new_images)
                    img_content = [{"type": "text", "text": f"[Visual content from tool results: {len(new_images)} image(s)]"}]
                    img_content.extend({"type": "image", "image": f"placeholder:{i}"} for i, _ in enumerate(new_images))
                    messages.append({"role": "user", "content": img_content})
                    _cap_images(messages, accumulated_images, config.max_images_per_rollout)
                else:
                    # Unimodal mode: keep the textual hint but do not feed images
                    # into the model, so the rollout stays in the text-only path.
                    messages.append({
                        "role": "user",
                        "content": f"[Visual content from tool results: {len(new_images)} image(s) omitted in unimodal mode]",
                    })

            samples.append(cur_sample)

        else:
            # max_steps reached: force a final response without tools
            if accumulated_images:
                final_prompt = state.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                final_prompt = sample.prompt

            final_sample = Sample(
                group_index=sample.group_index,
                index=sample.index,
                messages=deepcopy(messages),
                prompt=final_prompt,
                label=sample.label,
                status=Sample.Status.PENDING,
                metadata=sample.metadata,
                train_metadata={"tools": None},
                multimodal_inputs={"images": list(accumulated_images)} if accumulated_images else {},
            )
            final_sample.group_id = sample.index
            # messages[0] = {"role": "system", "content": system_prompt + "\n\n请直接回答，不要使用工具。"}
            messages[0] = {"role": "system", "content": system_prompt}
            final_sample.messages = deepcopy(messages)

            # Safety check: ensure image blocks in messages match accumulated_images
            _sync_images_and_messages(final_sample.messages, accumulated_images)
            if final_sample.multimodal_inputs:
                final_sample.multimodal_inputs["images"] = list(accumulated_images)
            elif accumulated_images:
                final_sample.multimodal_inputs = {"images": list(accumulated_images)}

            final_sample = await base_generate(args, final_sample, sampling_params)
            # Same tau2-bench-parity default as cur_sample above (see comment there).
            final_sample.teacher_log_probs = deepcopy(final_sample.rollout_log_probs)

            parsed = parse_qwen35_response(final_sample.response)
            final_sample.response_message = parsed
            messages.append({
                "role": "assistant",
                "content": final_sample.response.removesuffix(eos_token),
            })
            samples.append(final_sample)

        # Collect output artifacts from sandbox before teardown
        if needs_sandbox and sandbox is not None and samples:
            artifact_text = await _collect_output_artifacts(sandbox, task)
            if artifact_text:
                final_meta = samples[-1].metadata if samples[-1].metadata else {}
                final_meta["output_artifacts"] = artifact_text
                samples[-1].metadata = final_meta

        # Collect env_snapshot (rule-based grader verification) before teardown
        if needs_sandbox and sandbox is not None and samples:
            env_snapshot = await _collect_env_snapshot(sandbox, task, task_id)
            if env_snapshot:
                final_meta = samples[-1].metadata if samples[-1].metadata else {}
                final_meta["env_snapshot"] = env_snapshot
                samples[-1].metadata = final_meta

        # Collect screenshots for visual judge (after artifacts, before teardown)
        if needs_sandbox and sandbox is not None and samples:
            screenshots = await _collect_screenshots(sandbox, task, task_id)
            if screenshots:
                final_meta = samples[-1].metadata if samples[-1].metadata else {}
                final_meta["screenshots"] = screenshots
                samples[-1].metadata = final_meta

        if hasattr(dispatcher, "close"):
            dispatcher.close()

    # Guard: if no samples were produced (e.g. immediate timeout),
    # create a minimal fallback so group_reward always has valid messages.
    if not samples:
        fallback = Sample(
            group_index=sample.group_index,
            index=sample.index,
            messages=deepcopy(messages),
            prompt=sample.prompt,
            label=sample.label,
            tokens=[state.tokenizer.pad_token_id],
            reward=0.0,
            loss_mask=[],
            rollout_log_probs=[],
            teacher_log_probs=[],
        )
        fallback.group_id = sample.index
        fallback.response = ""
        samples.append(fallback)

    # Update final sample with complete conversation
    if samples:
        final = samples[-1]
        # Append the final step's own response (which was not in messages
        # because messages is deepcopy'd BEFORE base_generate)
        if final.response_message:
            final_msg = dict(final.response_message)
            if final.response:
                final_msg["raw"] = final.response.removesuffix(eos_token)
            final.messages.append(final_msg)
        elif final.response:
            final.messages.append(parse_qwen35_response(final.response))
        # Parse previous assistant messages into structured format, preserving raw
        for i, msg in enumerate(final.messages):
            if msg.get("role") == "assistant" and "tool_calls" not in msg:
                raw_field = msg.get("raw")
                parsed_msg = parse_qwen35_response(msg["content"])
                if raw_field is not None:
                    parsed_msg["raw"] = raw_field
                final.messages[i] = parsed_msg

    # NOTE: padding to a fixed length (max_steps + 1) is disabled.
    # It existed only to keep the flattened multi-turn sample count divisible by
    # global_batch_size (see data.py:get_data_iterator, num_steps_per_rollout =
    # num_local_samples // num_local_gbs). In practice a trajectory uses ~3 steps
    # while max_steps can be 20+, so >85% of the samples were zero-length pads
    # that only produced empty micro-batches (loss=0, grad_norm=0).
    # Divisibility is now handled by --use-dynamic-global-batch-size, which sets
    # global_batch_size to the actual sample count (num_steps_per_rollout == 1).
    # Prepended pads also made samples[0] a pad sample, which suppressed
    # train_data["metadata"] in _convert_samples_to_train_data.
    #
    # padding_num = (max_steps + 1) - len(samples)
    # if padding_num > 0:
    #     pad_samples = []
    #     for _ in range(padding_num):
    #         pad_sample = Sample(
    #             group_index=sample.group_index,
    #             index=sample.index,
    #             tokens=[state.tokenizer.pad_token_id],
    #             reward=0.0,
    #             loss_mask=[],
    #             rollout_log_probs=[],
    #         )
    #         pad_sample.group_id = sample.index
    #         pad_samples.append(pad_sample)
    #     samples = pad_samples + samples

    return samples
