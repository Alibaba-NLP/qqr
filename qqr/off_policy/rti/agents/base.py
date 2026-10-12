import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..client import OpenAIChatClient
from qqr.data.prompts.qwen import QwenPrompt


@dataclass
class AgentResult:
    raw: str
    parsed: Any = None
    ok: bool = True
    meta: Dict[str, Any] = None
    prompt: str = ""
    latency_ms: int = -1


_prompter = QwenPrompt()


class BaseAgent:
    name: str = "base"

    def __init__(self, client: OpenAIChatClient, enabled: bool = True):
        self.client = client
        self.enabled = enabled

    def parse_qwen3_output(self, raw: str, tools: list = None) -> dict:
        """Parse Qwen3/Qwen3.5 output using the shared QwenPrompt parser."""
        msg = _prompter.parse_assistant_content(raw, tools)
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            tool_calls.append({
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            })
        return {
            "reasoning": msg.get("reasoning_content", ""),
            "content": msg.get("content", ""),
            "tool_calls": tool_calls,
        }

    async def run(self, *, prompt: str, meta: Optional[dict] = None, return_message=False) -> AgentResult:
        if not self.enabled:
            return AgentResult(raw="", parsed=None, ok=True, meta={"disabled": True, **(meta or {})}, prompt=prompt)

        tools = (meta or {}).get("tools")
        t0 = time.time()
        if not tools:
            raw = await self.client.chat([{"role": "user", "content": prompt}], return_message=return_message)
        else:
            raw = await self.client.chat(
                [{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice="auto",
                return_message=return_message,
            )
        latency_ms = int((time.time() - t0) * 1000)

        return AgentResult(raw=raw, parsed=None, ok=bool(raw), meta=meta or {}, prompt=prompt, latency_ms=latency_ms)
