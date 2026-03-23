import asyncio
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


@registers.reward_model("single_elimination")
class SingleEliminationGroupRewardModel(GroupRewardModel):
    def __init__(self, llm_judge: LLMJudge):
        super().__init__()

        self.llm_judge = llm_judge

    async def compute(self, predictions: list[list[dict]], query: str) -> list[float]:
        group_size = len(predictions)

        players = [Player(idx=i) for i in range(group_size)]

        await self.compute_seeding_scores(players, predictions, query=query)

        bracket = self.get_seeded_bracket(players)
        champion, eliminated_history = await self.run_tournament(
            bracket, predictions, query=query
        )

        ranked_players = self.determine_final_ranks(champion, eliminated_history)
        group_rewards = self.calculate_group_rewards(ranked_players, group_size)
        return group_rewards

    async def compute_seeding_scores(
        self, players: list[Player], predictions: list[list[dict]], query: str
    ):
        """Runs a quick Anchor comparison (everyone vs Index 0) to establish an initial seeding score (avg_point)."""
        group_size = len(players)
        if group_size < 2:
            return

        pivot_idx = 0
        pivot_prediction = predictions[pivot_idx]
        pivot_scores = []

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
            score_other, score_pivot, metadata = task.result()
            idx = metadata["idx"]
            players[idx].points.append(score_other)
            pivot_scores.append(score_pivot)

        players[pivot_idx].points.append(statistics.mean(pivot_scores))

    async def run_tournament(
        self, bracket: list[Player], predictions: list[list[dict]], query: str
    ) -> tuple[Player | None, list[list[Player]]]:
        active_players = bracket[:]
        eliminated_history: list[list[Player]] = []

        while len(active_players) > 1:
            next_round_players = []
            round_losers = []

            # Create pairings based on current bracket order
            pairings = []
            i = 0
            while i < len(active_players):
                if i + 1 < len(active_players):
                    pairings.append((active_players[i], active_players[i + 1]))
                    i += 2
                else:
                    # Bye: Player advances automatically
                    next_round_players.append(active_players[i])
                    i += 1

            # Run comparisons
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for p1, p2 in pairings:
                    tasks.append(
                        tg.create_task(
                            self.llm_judge.bidirectional_compare(
                                predictions[p1.idx],
                                predictions[p2.idx],
                                query=query,
                                p1=p1,
                                p2=p2,
                            )
                        )
                    )

            # Process results
            for task in tasks:
                score_1, score_2, metadata = task.result()
                p1, p2 = metadata["p1"], metadata["p2"]

                p1.points.append(score_1)
                p2.points.append(score_2)

                if score_1 >= score_2:
                    next_round_players.append(p1)
                    round_losers.append(p2)
                else:
                    next_round_players.append(p2)
                    round_losers.append(p1)

            if round_losers:
                eliminated_history.append(round_losers)

            active_players = next_round_players

        champion = active_players[0] if active_players else None
        return champion, eliminated_history

    def get_seeded_bracket(self, players: list[Player]) -> list[Player]:
        """Arranges players so high seeds don't meet early."""
        group_size = len(players)
        sorted_players = sorted(players, key=lambda p: p.avg_point, reverse=True)

        # Find next power of 2
        power = 1
        while power < group_size:
            power *= 2

        bracket_indices = [0]
        current_count = 1
        while current_count < power:
            next_bracket = []
            for i in bracket_indices:
                next_bracket.append(i)
                next_bracket.append(2 * current_count - 1 - i)
            bracket_indices = next_bracket
            current_count *= 2

        # Filtering out "byes" (idx >= group_size)
        final_bracket = [
            sorted_players[idx] for idx in bracket_indices if idx < group_size
        ]
        return final_bracket

    def determine_final_ranks(
        self, champion: Player | None, eliminated_history: list[list[Player]]
    ) -> list[Player]:
        ranked_players = []
        if champion:
            ranked_players = [champion]

        # Reverse history: Final Loser -> Semifinal Losers -> ...
        for group in reversed(eliminated_history):
            # Sort losers in the same round by their average points
            group.sort(key=lambda p: p.avg_point, reverse=True)
            ranked_players.extend(group)

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
