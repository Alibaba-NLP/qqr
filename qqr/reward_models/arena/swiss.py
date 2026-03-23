import asyncio
import math
import random
from dataclasses import dataclass, field

import torch

from qqr import registers
from qqr.schemas import GroupRewardModel, LLMJudge


@dataclass
class Player:
    idx: int
    points: float = 0.0
    opponents: set[int] = field(default_factory=set)
    buchholz: float = 0.0


@registers.reward_model("swiss")
class SwissSystemGroupRewardModel(GroupRewardModel):
    def __init__(self, llm_judge: LLMJudge, max_num_rounds: int | None = None):
        super().__init__()

        self.llm_judge = llm_judge
        self.max_num_rounds = max_num_rounds

    async def compute(self, predictions: list[list[dict]], query: str) -> list[float]:
        group_size = len(predictions)

        num_rounds = self.get_num_rounds(group_size)
        players = [Player(idx=i) for i in range(group_size)]
        for _ in range(num_rounds):
            pairings, bye_player_idx = self.create_pairings(players)

            tasks = []
            async with asyncio.TaskGroup() as tg:
                for i, j in pairings:
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
                    players[i].points += 1.0
                elif score_j > score_i:
                    players[j].points += 1.0
                else:
                    players[i].points += 0.5
                    players[j].points += 0.5

                players[i].opponents.add(j)
                players[j].opponents.add(i)

            if bye_player_idx is not None:
                players[bye_player_idx].points += 1.0

        self.calculate_buchholz(players)
        group_rewards = self.calculate_group_rewards(players, group_size)

        return group_rewards

    def get_num_rounds(self, group_size: int) -> int:
        if self.max_num_rounds is not None and self.max_num_rounds > 0:
            num_rounds = self.max_num_rounds
        else:
            num_rounds = math.ceil(math.log2(group_size))

        num_rounds = min(num_rounds, group_size - 1)
        return num_rounds

    def create_pairings(self, players: list[Player]) -> tuple[list, int | None]:
        random.shuffle(players)
        players_sorted = sorted(players, key=lambda p: p.points, reverse=True)

        unpaired = players_sorted[:]
        pairings = []
        bye_player_idx = None

        if len(unpaired) % 2 != 0:
            bye_player_idx = unpaired.pop(-1).idx

        processed = [False] * len(unpaired)
        for i in range(len(unpaired)):
            if processed[i]:
                continue

            p1 = unpaired[i]
            found_opponent = False
            for j in range(i + 1, len(unpaired)):
                if not processed[j] and unpaired[j].idx not in p1.opponents:
                    p2 = unpaired[j]
                    pairings.append((p1.idx, p2.idx))
                    processed[i] = processed[j] = True
                    found_opponent = True
                    break

            if not found_opponent:
                for j in range(i + 1, len(unpaired)):
                    if not processed[j]:
                        p2 = unpaired[j]
                        pairings.append((p1.idx, p2.idx))
                        processed[i] = processed[j] = True
                        break

        return pairings, bye_player_idx

    def calculate_buchholz(self, players: list[Player]):
        for p in players:
            p.buchholz = sum(players[opp_idx].points for opp_idx in p.opponents)

    def calculate_group_rewards(
        self, players: list[Player], group_size: int
    ) -> list[float]:
        group_rewards = [0.0] * group_size

        ranked_players = sorted(
            players, key=lambda p: (p.points, p.buchholz), reverse=True
        )

        i = 0
        while i < group_size:
            j = i
            while (
                j + 1 < group_size
                and ranked_players[j + 1].points == ranked_players[i].points
                and ranked_players[j + 1].buchholz == ranked_players[i].buchholz
            ):
                j += 1

            sum_rewards = sum(
                (group_size - (k + 1)) / (group_size - 1) for k in range(i, j + 1)
            )
            avg_reward = sum_rewards / (j - i + 1)

            for k in range(i, j + 1):
                p_idx = ranked_players[k].idx
                group_rewards[p_idx] = avg_reward

            i = j + 1

        group_rewards = torch.tensor(group_rewards, dtype=torch.float)
        mean = group_rewards.mean(dim=-1, keepdim=True)
        std = group_rewards.std(dim=-1, keepdim=True)
        group_rewards = (group_rewards - mean) / (std + 1e-6)
        group_rewards = group_rewards.flatten().tolist()

        return group_rewards
