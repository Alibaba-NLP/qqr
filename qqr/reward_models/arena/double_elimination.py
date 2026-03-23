import asyncio
import random
import statistics
from dataclasses import dataclass, field

import torch

from qqr import registers
from qqr.schemas import GroupRewardModel, LLMJudge


@dataclass
class Player:
    idx: int
    points: list[float] = field(default_factory=list)

    @property
    def avg_point(self) -> float:
        return statistics.mean(self.points) if self.points else 0.0


@registers.reward_model("double_elimination")
class DoubleEliminationGroupRewardModel(GroupRewardModel):
    def __init__(self, llm_judge: LLMJudge):
        super().__init__()

        self.llm_judge = llm_judge

    async def compute(self, predictions: list[list[dict]], query: str) -> list[float]:
        group_size = len(predictions)

        players = [Player(idx=i) for i in range(group_size)]

        wb_champion, wb_drops_schedule = await self.run_winners_bracket(
            players, predictions, query=query
        )
        lb_champion, lb_eliminated_history = await self.run_losers_bracket(
            wb_drops_schedule, predictions, query=query
        )
        grand_winner, grand_loser = await self.run_grand_final(
            wb_champion, lb_champion, predictions, query=query
        )

        ranked_players = self.determine_final_ranks(
            players, grand_winner, grand_loser, lb_eliminated_history
        )
        group_rewards = self.calculate_group_rewards(ranked_players, group_size)
        return group_rewards

    async def play_round(
        self, players: list[Player], predictions: list[list[dict]], query: str
    ) -> tuple[list[Player], list[Player]]:
        pairings, byes = self.create_pairings(players)

        winners = byes[:]
        losers = []
        tasks = []
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    self.llm_judge.bidirectional_compare(
                        predictions[p1.idx],
                        predictions[p2.idx],
                        query=query,
                        p1=p1,
                        p2=p2,
                    )
                )
                for p1, p2 in pairings
            ]

        for task in tasks:
            score_1, score_2, metadata = task.result()
            p1, p2 = metadata["p1"], metadata["p2"]
            p1.points.append(score_1)
            p2.points.append(score_2)

            if score_1 >= score_2:
                winners.append(p1)
                losers.append(p2)
            else:
                winners.append(p2)
                losers.append(p1)

        return winners, losers

    async def run_winners_bracket(
        self, players: list[Player], predictions: list[list[dict]], query: str
    ) -> tuple[Player, list[list[Player]]]:
        active_players: list[Player] = players[:]
        drops_schedule: list[list[Player]] = []

        while len(active_players) > 1:
            winners, losers = await self.play_round(
                active_players, predictions, query=query
            )
            active_players = winners
            if losers:
                drops_schedule.append(losers)

        wb_champion = active_players[0] if active_players else None
        return wb_champion, drops_schedule

    async def run_losers_bracket(
        self, wb_drops: list[list[Player]], predictions: list[list[dict]], query: str
    ) -> tuple[Player | None, list[list[Player]]]:
        active_players: list[Player] = []
        eliminated_history: list[list[Player]] = []

        for dropped_players in wb_drops:
            active_players.extend(dropped_players)

            if len(active_players) >= 2:
                winners, losers = await self.play_round(
                    active_players, predictions, query=query
                )
                active_players = winners
                if losers:
                    eliminated_history.append(losers)

        while len(active_players) > 1:
            winners, losers = await self.play_round(
                active_players, predictions, query=query
            )
            active_players = winners
            if losers:
                eliminated_history.append(losers)

        lb_champion = active_players[0] if active_players else None
        return lb_champion, eliminated_history

    async def run_grand_final(
        self,
        wb_champ: Player | None,
        lb_champ: Player | None,
        predictions: list[list[dict]],
        query: str,
    ) -> tuple[Player, Player | None]:
        grand_winner = wb_champ
        grand_loser = lb_champ

        if wb_champ and lb_champ and wb_champ.idx != lb_champ.idx:
            winners, losers = await self.play_round(
                [wb_champ, lb_champ], predictions, query=query
            )
            grand_winner = winners[0] if winners else None
            grand_loser = losers[0] if losers else None

        return grand_winner, grand_loser

    def create_pairings(
        self, players: list[Player]
    ) -> tuple[list[tuple[Player, Player]], list[Player]]:
        pool = players[:]
        random.shuffle(pool)

        pairings = []
        while len(pool) >= 2:
            pairings.append((pool.pop(), pool.pop()))

        byes = pool
        return pairings, byes

    def determine_final_ranks(
        self,
        players: list[Player],
        grand_winner: Player | None,
        grand_loser: Player | None,
        lb_elim_history: list[list[Player]],
    ) -> list[Player]:
        ranked_players: list[Player] = []

        if grand_winner:
            ranked_players.append(grand_winner)
        if grand_loser:
            ranked_players.append(grand_loser)

        for losers_group in reversed(lb_elim_history):
            losers_group.sort(key=lambda p: p.avg_point, reverse=True)
            ranked_players.extend(losers_group)

        ranked_ids = {p.idx for p in ranked_players}
        leftovers = [p for p in players if p.idx not in ranked_ids]
        if leftovers:
            leftovers.sort(key=lambda p: p.avg_point, reverse=True)
            ranked_players.extend(leftovers)

        return ranked_players

    def calculate_group_rewards(
        self, ranked_players: list[Player], group_size: int
    ) -> list[float]:
        group_rewards = [0.0] * group_size

        for rank_idx, player in enumerate(ranked_players):
            reward = 1.0 - (rank_idx / (group_size - 1))
            group_rewards[player.idx] = reward

        group_rewards = torch.tensor(group_rewards, dtype=torch.float)
        mean = group_rewards.mean(dim=-1, keepdim=True)
        std = group_rewards.std(dim=-1, keepdim=True)
        group_rewards = (group_rewards - mean) / (std + 1e-6)
        group_rewards = group_rewards.flatten().tolist()

        return group_rewards
