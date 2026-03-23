import asyncio

import numpy as np
import pandas as pd
import torch

from qqr import registers
from qqr.schemas import GroupRewardModel, LLMJudge


@registers.reward_model("anchor")
class AnchorBasedRankingGroupRewardModel(GroupRewardModel):
    def __init__(self, llm_judge: LLMJudge):
        super().__init__()

        self.llm_judge = llm_judge

    async def compute(self, predictions: list[list[dict]], query: str) -> list[float]:
        group_size = len(predictions)

        pivot_idx = 0
        pivot_prediction = predictions[pivot_idx]
        pivot_scores = [5.0] * group_size
        other_scores = [5.0] * group_size
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for idx in range(1, group_size):
                task = tg.create_task(
                    self.llm_judge.bidirectional_compare(
                        predictions[idx], pivot_prediction, query=query, idx=idx
                    )
                )
                tasks.append(task)

        for task in tasks:
            other_score, pivot_score, metadata = task.result()
            idx = metadata["idx"]
            other_scores[idx] = other_score
            pivot_scores[idx] = pivot_score

        pivot_scores = pivot_scores[1:]
        pivot_mean_score = np.mean(pivot_scores)
        scores = [pivot_mean_score] + other_scores[1:]
        ranks = pd.Series(scores).rank(method="min", ascending=False).tolist()
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
