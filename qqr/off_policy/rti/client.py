import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List

import httpx
import re

@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_new_tokens: int = 800
    timeout: int = 500
    max_retries: int = 10
    retry_base_backoff: float = 10.0
    retry_max_backoff: float = 300.0


class OpenAIChatClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = httpx.AsyncClient(timeout=cfg.timeout)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        return_message: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        - 默认返回 assistant.content (str)，兼容你现有逻辑
        - return_message=True 时返回完整 assistant message(dict)，其中可能包含 tool_calls
        - tools: OpenAI tools schema: list[{"type":"function","function":{...}}]
        - tool_choice: "auto"/"none"/{"type":"function","function":{"name":"xxx"}}
        """
        if not self.cfg.base_url or not self.cfg.api_key or not self.cfg.model:
            return {} if return_message else ""

        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        data: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_new_tokens": self.cfg.max_new_tokens,
            "logprobs": True,
        }
        if tools is not None:
            data["tools"] = tools
        if tool_choice is not None:
            data["tool_choice"] = tool_choice
        if extra:
            data.update(extra)

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                r = await self._client.post(
                    f"{self.cfg.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                )

                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"Retryable status: {r.status_code}",
                        request=r.request,
                        response=r,
                    )
                    
                r.raise_for_status()
                j = r.json()
                msg = (j.get("choices", [{}])[0].get("message") or {})
                lp_content = j["choices"][0]["logprobs"]["content"]
                msg['tokens'] = [x["token"] for x in lp_content]
                msg['token_logprobs'] = [x["logprob"] for x in lp_content]

                # Extract content and apply Qwen-specific cleaning
                content = msg.get("content") or ""
                
                # Only clean for Qwen3.5-397B-A17B model
                if content and '</think>' in content:
                    # Remove everything before and including </think>
                    if '</think>' in content:
                        reasoning_content = content.split('</think>', 1)[0]
                        content = content.split('</think>', 1)[1]
                        # Optional: strip leading whitespace/newlines
                        content = content.lstrip('\n').lstrip()
                        msg['content'] = content
                        msg['reasoning'] = reasoning_content

                if return_message:
                    return msg

                ans = (msg.get("content") or "").strip()
                if ans:
                    return ans

                # content 为空但 tool_calls 存在：对“只需要 content”的旧逻辑而言仍算 empty
                last_exc = RuntimeError("Empty content from LLM (maybe tool_calls only).")

            except Exception as e:
                last_exc = e

            if attempt < self.cfg.max_retries:
                backoff = min(self.cfg.retry_max_backoff, self.cfg.retry_base_backoff * (2 ** (attempt - 1)))
                backoff = backoff * (0.8 + 0.4 * random.random())
                await asyncio.sleep(backoff)

        return {} if return_message else ""
