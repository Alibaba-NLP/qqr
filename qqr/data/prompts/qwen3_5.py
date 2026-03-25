import ast
import json
import logging
import re
from typing import Any

from qqr import registers
from qqr.schemas import Prompt

logger = logging.getLogger(__name__)


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/qwen3coder_tool_parser.py
@registers.prompt("qwen3.5")
class Qwen3_5Prompt(Prompt):
    eos_token = "<|im_end|>"
    bot_token = "<tool_call>"
    eot_token = "</tool_call>"
    think_start_token = "<think>"
    think_end_token = "</think>"
    function_start_token = "<function="
    function_end_token = "</function>"
    parameter_start_token = "<parameter="
    parameter_end_token = "</parameter>"

    think_pattern = re.compile(r"<think>(.*?)</think>", re.S)
    tool_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
    tool_call_pattern = re.compile(
        r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
    )
    function_pattern = re.compile(
        r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
    )
    parameter_pattern = re.compile(
        r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
        re.DOTALL,
    )

    def _get_arguments_config(self, func_name: str, tools: list[dict] | None) -> dict:
        """Extract argument configuration for a function from tools schema."""
        if tools is None:
            return {}
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            if function.get("name") != func_name:
                continue
            params = function.get("parameters", {})
            if isinstance(params, dict) and "properties" in params:
                return params["properties"]
            elif isinstance(params, dict):
                return params
            else:
                return {}
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.debug(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the string value.",
                    param_name,
                    func_name,
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        elif (
            isinstance(param_config[param_name], dict)
            and "anyOf" in param_config[param_name]
        ):
            param_type = "object"
        else:
            param_type = "string"

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not an "
                    "integer in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a float "
                    "in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a boolean "
                    "(`true` or `false`) in tool '%s', degenerating to false.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.debug(
                        "Parsed value '%s' of parameter '%s' cannot be "
                        "parsed with json.loads in tool '%s', will try "
                        "other methods to parse it.",
                        param_value,
                        param_name,
                        func_name,
                    )
            try:
                param_value = ast.literal_eval(param_value)
            except (ValueError, SyntaxError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "converted via Python `ast.literal_eval()` in tool "
                    "'%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[dict] | None = None
    ) -> dict | None:
        """Parse XML-style function call."""
        # Extract function name
        end_index = function_call_str.find(">")
        if end_index == -1:
            return None

        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}

        for match_text in self.parameter_pattern.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])

            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )

        return {
            "name": function_name,
            "arguments": json.dumps(param_dict, ensure_ascii=False),
        }

    def _get_function_calls(self, model_output: str) -> list[str]:
        """Find all function calls in model output."""
        matched_ranges = self.tool_call_pattern.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.function_pattern.findall(tool_call))

        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls

    def parse_assistant_content(
        self, assistant_content: str, tools: list[dict] | None = None, **kwargs
    ) -> dict:
        message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [],
        }

        if (
            self.think_end_token in assistant_content
            and not assistant_content.startswith(self.think_start_token)
        ):
            assistant_content = self.think_start_token + assistant_content

        if self.think_start_token in assistant_content:
            think_match = self.think_pattern.search(assistant_content)
            if think_match:
                message["reasoning_content"] = think_match.group(1).strip()
                assistant_content = self.think_pattern.sub("", assistant_content)

        if self.function_start_token in assistant_content:
            # Use XML-style parsing
            try:
                function_calls = self._get_function_calls(assistant_content)

                for func_idx, func_str in enumerate(function_calls):
                    parsed = self._parse_xml_function_call(func_str, tools)
                    if parsed:
                        message["tool_calls"].append(
                            {
                                "id": f"call_{func_idx + 1}",
                                "type": "function",
                                "function": {
                                    "name": parsed["name"],
                                    "arguments": parsed["arguments"],
                                },
                            }
                        )
            except Exception:
                logger.exception("Error in extracting tool call from response.")

            # Extract content before tool calls
            content_index = assistant_content.find(self.bot_token)
            if content_index == -1:
                content_index = assistant_content.find(self.function_start_token)
            if content_index >= 0:
                assistant_content = assistant_content[:content_index]

        message["content"] = assistant_content.removesuffix(self.eos_token).strip()

        return message
