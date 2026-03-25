import json
import logging
import re

from qqr import registers
from qqr.schemas import Prompt

logger = logging.getLogger(__name__)


@registers.prompt("qwen3")
class Qwen3Prompt(Prompt):
    eos_token = "<|im_end|>"
    bot_token = "<tool_call>"
    eot_token = "</tool_call>"
    think_start_token = "<think>"
    think_end_token = "</think>"

    think_pattern = re.compile(r"<think>(.*?)</think>", re.S)
    tool_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

    def parse_assistant_content(self, assistant_content: str, **kwargs) -> dict:
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

        if self.bot_token in assistant_content:
            tool_matches = self.tool_pattern.findall(assistant_content)

            for func_idx, func_json_str in enumerate(tool_matches):
                func_json_str = func_json_str.strip()
                try:
                    tool_call = json.loads(func_json_str)

                    func_name = tool_call.get("name")
                    func_args = tool_call.get("arguments", {})

                    if isinstance(func_args, (dict, list)):
                        func_args_str = json.dumps(func_args, ensure_ascii=False)
                    else:
                        func_args_str = str(func_args)

                    message["tool_calls"].append(
                        {
                            "id": f"call_{func_idx + 1}",
                            "type": "function",
                            "function": {"name": func_name, "arguments": func_args_str},
                        }
                    )
                except Exception:
                    logger.exception("Failed to parse tool call from response.")
                    continue

            assistant_content = self.tool_pattern.sub("", assistant_content)

        message["content"] = assistant_content.removesuffix(self.eos_token).strip()

        return message
