import asyncio

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from qqr import registers
from qqr.schemas import LLM


@registers.llm("openai")
class OpenAI(LLM):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        concurrency_limit: int = 10,
        timeout: float = 60.0,
        max_retries: int = 10,
    ):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def __call__(self, messages, **kwargs) -> ChatCompletion:
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                messages=messages, model=self.model, **kwargs
            )

        return response
