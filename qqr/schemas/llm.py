import asyncio

import httpx


class LLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        concurrency_limit: int = 10,
        timeout: float = 60.0,
    ):
        self.model = model
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            base_url=base_url,
        )
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def __call__(self, url: str = "/chat/completions", **kwargs) -> dict:
        payload = {"model": self.model, **kwargs}

        async with self.semaphore:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()

        return response.json()
