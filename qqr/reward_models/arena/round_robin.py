import asyncio
import itertools

import pandas as pd
import torch

from qqr import registers
from qqr.schemas import GroupRewardModel, LLMJudge


@registers.reward_model("round_robin")
class RoundRobinGroupRewardModel(GroupRewardModel):
    def __init__(self, llm_judge: LLMJudge):
        super().__init__()

        self.llm_judge = llm_judge

    async def compute(self, predictions: list[list[dict]], query: str) -> list[float]:
        group_size = len(predictions)

        wins = [0.0] * group_size
        pairs = list(itertools.combinations(range(group_size), 2))
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for i, j in pairs:
                task = tg.create_task(
                    self.llm_judge.bidirectional_compare(
                        predictions[i], predictions[j], query=query, i=i, j=j
                    )
                )
                tasks.append(task)

        for task in tasks:
            score_i, score_j, metadata = task.result()
            i, j = metadata["i"], metadata["j"]

            if score_i > score_j:
                wins[i] += 1.0
            elif score_j > score_i:
                wins[j] += 1.0
            else:
                wins[i] += 0.5
                wins[j] += 0.5

        ranks = pd.Series(wins).rank(method="min", ascending=False).tolist()
        max_rank = max(ranks)

        if max_rank == 1:
            group_rewards = [0.0] * group_size
        else:
            group_rewards = [(max_rank - r) / (max_rank - 1) for r in ranks]

        group_rewards = torch.tensor(group_rewards, dtype=torch.float)
        mean = group_rewards.mean(dim=-1, keepdim=True)
        std = group_rewards.std(dim=-1, keepdim=True)
        group_rewards = (group_rewards - mean) / (std + 1e-6)
        group_rewards = group_rewards.flatten().tolist()

        return group_rewards
