"""
Qwen-family model output parsers.

Provides:
  - Content-driven format detection (think block, tool call format)
  - Common parsing pipeline (extract thinking → extract tool calls → remainder as content)
  - Schema-based argument type casting

`QwenPrompt` auto-detects the tool call format per block:
  - JSON format (Qwen3):        <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  - XML-param format (Qwen3.5): <tool_call><function=name>...</function></tool_call>

Note: this module is self-contained on purpose. It does not depend on
`qqr.data.prompts.qwen3` / `qqr.data.prompts.qwen3_5`, which are registry-based
prompt classes (`@registers.prompt`), so it can be imported from any context
(e.g. the RTI inspector utilities).
"""

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# XML-parameter tool call patterns (Qwen3.5)
_XML_FUNC_PATTERN = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
_XML_PARAM_PATTERN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)

EOS_TOKEN = "<|im_end|>"


def _cast_arguments(func_name: str, args: dict, tool_schema: dict) -> dict:
    """Cast tool arguments to schema-declared types."""
    if func_name not in tool_schema:
        return args
    schema = tool_schema[func_name]
    new_args = {}
    for k, v in args.items():
        if k not in schema:
            new_args[k] = v
            continue
        t = schema[k].get("type")
        try:
            if t == "integer":
                new_args[k] = int(v)
            elif t == "number":
                new_args[k] = float(v)
            elif t == "boolean":
                new_args[k] = v.lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v)
            elif t == "string":
                new_args[k] = str(v)
            elif t == "array":
                if isinstance(v, list):
                    new_args[k] = v
                elif isinstance(v, str):
                    try:
                        new_args[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        new_args[k] = v
                else:
                    new_args[k] = v
            elif t == "object":
                if isinstance(v, dict):
                    new_args[k] = v
                elif isinstance(v, str):
                    try:
                        new_args[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        new_args[k] = v
                else:
                    new_args[k] = v
            else:
                new_args[k] = v
        except Exception:
            new_args[k] = v
    return new_args


def _gen_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


class QwenPromptBase(ABC):
    """Abstract base class for Qwen model output parsing.

    Subclasses must implement `_parse_tool_block` for format-specific tool call parsing.
    The base class handles:
      - <think> block detection (including auto-fix for missing opening tag)
      - <tool_call> block extraction
      - Schema-based argument casting
      - EOS token stripping
    """

    @abstractmethod
    def _parse_tool_block(self, func_block: str, tools: list | None = None) -> dict | None:
        """Parse a single <tool_call>...</tool_call> inner content into a tool call dict.

        Returns dict with keys: id, type, function.{name, arguments}
        Or None if parsing fails.
        """
        ...

    def parse_assistant_content(self, content: str, tools: list = None) -> dict:
        """Parse raw assistant output into structured message dict."""
        message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [],
            "raw": content,
        }

        # Auto-fix: missing <think> when </think> is present
        if "</think>" in content and "<think>" not in content:
            content = "<think>" + content

        # Extract thinking block
        if "<think>" in content:
            think_match = _THINK_PATTERN.search(content)
            if think_match:
                message["reasoning_content"] = think_match.group(1).strip()
                content = _THINK_PATTERN.sub("", content)

        # Extract tool calls
        if "<tool_call>" in content:
            tool_matches = _TOOL_PATTERN.findall(content)
            tool_schema = None
            if tools is not None:
                tool_schema = {
                    t["function"]["name"]: t["function"]["parameters"]["properties"]
                    for t in tools
                }

            # Track which tool_call blocks were successfully parsed
            successfully_parsed = []
            for func_block in tool_matches:
                tc = self._parse_tool_block(func_block, tools)
                if tc is None:
                    # Parsing failed - keep this block in content
                    continue
                if tool_schema:
                    fn = tc["function"]["name"]
                    args = tc["function"]["arguments"]
                    tc["function"]["arguments"] = _cast_arguments(fn, args, tool_schema)
                message["tool_calls"].append(tc)
                successfully_parsed.append(func_block)

            # Only remove successfully parsed tool_call blocks
            # Failed parses remain in content so model can see its broken output
            for func_block in successfully_parsed:
                full_block = "<tool_call>" + func_block + "</tool_call>"
                content = content.replace(full_block, "", 1)
        message["content"] = content.removesuffix(EOS_TOKEN).strip()
        return message

    def parse_response(self, response: str, tools: list = None, *, tokenizer=None) -> dict:
        """Parse model response with auto EOS stripping.

        Content-driven detection handles all format variations automatically.
        """
        msg = self.parse_assistant_content(response, tools)
        if tokenizer:
            eos = getattr(tokenizer, 'eos_token', '') or ''
            if eos:
                msg['content'] = msg['content'].removesuffix(eos)
                msg['reasoning_content'] = msg['reasoning_content'].removesuffix(eos)
                msg['raw'] = msg['raw'].removesuffix(eos)
        return msg


class Qwen3Prompt(QwenPromptBase):
    """Parser for Qwen3 models (JSON tool call format)."""

    def _parse_tool_block(self, func_block: str, tools: list | None = None) -> dict | None:
        stripped = func_block.strip()
        try:
            tool_call = json.loads(stripped)
        except json.JSONDecodeError:
            return None

        func_name = tool_call.get("name")
        if not func_name or not isinstance(func_name, str):
            return None

        func_args = tool_call.get("arguments", {})
        if func_args is None:
            func_args = {}
        elif isinstance(func_args, str):
            try:
                func_args = json.loads(func_args)
            except (json.JSONDecodeError, ValueError):
                logger.debug(
                    f"Tool call arguments parse error for {func_name}: {func_args[:100]}"
                )
                return None
        elif not isinstance(func_args, dict):
            logger.debug(
                f"Tool call arguments type error for {func_name}: {type(func_args).__name__}"
            )
            return None

        existing_id = tool_call.get("id")
        call_id = existing_id if (isinstance(existing_id, str) and existing_id.startswith("call_")) else _gen_call_id()

        return {
            "id": call_id,
            "type": "function",
            "function": {"name": func_name, "arguments": func_args},
        }


class Qwen35Prompt(QwenPromptBase):
    """Parser for Qwen3.5 models (XML-parameter tool call format)."""

    def _parse_tool_block(self, func_block: str, tools: list | None = None) -> dict | None:
        stripped = func_block.strip()
        func_match = _XML_FUNC_PATTERN.search(stripped)
        if not func_match:
            return None

        func_name = func_match.group(1).strip()
        param_block = func_match.group(2)
        func_args: dict = {}

        for param_match in _XML_PARAM_PATTERN.finditer(param_block):
            key = param_match.group(1).strip()
            raw_val = param_match.group(2).strip()
            try:
                func_args[key] = json.loads(raw_val)
            except (json.JSONDecodeError, ValueError):
                func_args[key] = raw_val

        # If param_block has content but zero parameters were parsed,
        # this is a format error (model used wrong XML syntax).
        # Return None to signal parsing failure, not a valid empty tool call.
        if param_block.strip() and not func_args:
            logger.debug(
                f"Tool call format error: {func_name} has content in param_block "
                f"but no parameters parsed. Raw: {param_block[:100]}"
            )
            return None

        return {
            "id": _gen_call_id(),
            "type": "function",
            "function": {"name": func_name, "arguments": func_args},
        }


_JSON_PARSER = Qwen3Prompt()
_XML_PARSER = Qwen35Prompt()


class QwenPrompt(QwenPromptBase):
    """Auto-detecting parser that delegates to JSON or XML parsing based on content.

    Use this when you don't know the model version upfront — it inspects each
    tool_call block and picks the right strategy.
    """

    def _parse_tool_block(self, func_block: str, tools: list | None = None) -> dict | None:
        stripped = func_block.strip()
        if stripped.startswith("<function="):
            return _XML_PARSER._parse_tool_block(stripped, tools)
        result = _JSON_PARSER._parse_tool_block(stripped, tools)
        if result is None:
            result = _XML_PARSER._parse_tool_block(stripped, tools)
        return result
