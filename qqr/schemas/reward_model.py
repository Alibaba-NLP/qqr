from abc import ABC, abstractmethod

from .llm import LLM


class RewardModel(ABC):
    async def __call__(self, *args, **kwargs) -> float | dict[str, float]:
        return await self.compute(*args, **kwargs)

    @abstractmethod
    async def compute(
        self, prediction, reference=None, *args, **kwargs
    ) -> float | dict[str, float]: ...


class GroupRewardModel(ABC):
    async def __call__(self, *args, **kwargs) -> list[float] | list[dict[str, float]]:
        return await self.compute(*args, **kwargs)

    @abstractmethod
    async def compute(
        self, predictions: list, reference=None, *args, **kwargs
    ) -> list[float] | list[dict[str, float]]: ...


class LLMRewardModel(RewardModel):
    def __init__(self, llm: LLM):
        self.llm = llm
