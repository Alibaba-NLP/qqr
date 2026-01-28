"""
Travel Agent Reward Model (SerpApi version)
"""
import asyncio
import logging
import re
from argparse import Namespace

from openai import AsyncOpenAI

from qqr.reward_models import get_reward_model
from qqr.schemas import LLMJudge, Sample

from . import config

logger = logging.getLogger(__name__)


class TravelLLMJudge(LLMJudge):
    def __init__(self):
        self.system_prompt = config.llm_judge_system_prompt
        self.model = config.llm_judge_model
        self._client = None

        self.concurrency_limit = config.llm_judge_concurrency_limit
        self._semaphore: asyncio.Semaphore | None = None

        self.score_a_pattern = re.compile(
            r'"combined_scores"\s*:\s*\{[^{}]*?"Agent_A"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            re.S | re.I,
        )
        self.score_b_pattern = re.compile(
            r'"combined_scores"\s*:\s*\{[^{}]*?"Agent_B"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            re.S | re.I,
        )
        self.winner_pattern = re.compile(
            r'"winner"\s*:\s*"(?P<winner>Agent_A|Agent_B|Tie)"', re.I
        )

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=config.llm_judge_api_key,
                base_url=config.llm_judge_base_url,
                timeout=60,
                max_retries=10,
            )
        return self._client

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """
        Lazy-initialized semaphore that binds to the current running Event Loop.

        Using lazy initialization prevents "Future attached to a different loop" errors
        when the server instance persists across multiple asyncio.run() calls or
        event loop restarts.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        return self._semaphore

    async def compare(
        self, messages_a: list[dict], messages_b: list[dict], query: str
    ) -> tuple[float, float]:
        trajectory_a, answer_a = self.process_messages(messages_a)
        trajectory_b, answer_b = self.process_messages(messages_b)

        prompt = f"""<USER_QUERY>\n{query}\n</USER_QUERY>\n\n<PATH_A>\n{trajectory_a}\n</PATH_A>\n\n<PATH_B>\n{trajectory_b}\n</PATH_B>\n\n<Answer_A>\n{answer_a}\n</Answer_A>\n\n<Answer_B>\n{answer_b}\n</Answer_B>"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        score_a, score_b = 5.0, 5.0
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    messages=messages, model=self.model, temperature=0.0
                )

            score_a, score_b = self.get_judge_scores(
                response.choices[0].message.content
            )

        except Exception as e:
            logger.warning(f"[LLMJudge] Failed to get result: {e}")

        return score_a, score_b

    async def bidirectional_compare(
        self, messages_a: list[dict], messages_b: list[dict], query: str, **kwargs
    ) -> tuple[float, float, dict]:
        results = await asyncio.gather(
            self.compare(messages_a, messages_b, query=query),
            self.compare(messages_b, messages_a, query=query),
        )

        score_a = results[0][0] + results[1][1]
        score_b = results[0][1] + results[1][0]

        return score_a, score_b, kwargs

    def process_messages(self, messages: list[dict]) -> tuple[list[dict], str]:
        step_idx = 0
        trajectory = []
        for message in messages[:-1]:
            if message["role"] != "assistant":
                continue

            step_idx += 1
            trajectory.append(
                {
                    "step": step_idx,
                    "reasoning_content": message.get("reasoning_content", ""),
                    "tool_calls": message.get("tool_calls", ""),
                }
            )

        answer = "未回复"
        if messages[-1]["role"] == "assistant":
            answer = messages[-1].get("content") or answer

        return trajectory, answer

    def get_judge_scores(self, response: str) -> tuple[float, float]:
        score_a, score_b = 5.0, 5.0

        try:
            match_a = self.score_a_pattern.search(response)
            match_b = self.score_b_pattern.search(response)

            if match_a and match_b:
                score_a = float(match_a.group(1))
                score_b = float(match_b.group(1))

        except:
            pass

        return score_a, score_b


llm_judge = TravelLLMJudge()
group_reward_model = get_reward_model(config.group_reward_model_name)(llm_judge)


async def eval_reward(args: Namespace, sample: Sample, **kwargs):
    prediction = sample.messages
    reference = sample.label
    if isinstance(sample.prompt, str):
        query = sample.prompt
    else:
        query = sample.prompt[-1]["content"]

    pred_score, ref_score, metadata = await llm_judge.bidirectional_compare(
        prediction, reference, query=query
    )

    if pred_score > ref_score:
        sample.reward = 1
    elif pred_score < ref_score:
        sample.reward = 0
    else:
        sample.reward = 0.5


async def group_reward(args: Namespace, group: list[list[Sample]], **kwargs):
    if len(group) <= 1:
        raise ValueError("group size must be greater than 1")

    predictions = [g[-1].messages for g in group]
    if isinstance(group[0][0].prompt, str):
        query = group[0][0].prompt
    else:
        query = group[0][0].prompt[-1]["content"]

    group_rewards = await group_reward_model(predictions=predictions, query=query)

    for idx in range(len(group)):
        for sample in group[idx]:
            sample.reward = group_rewards[idx]


def reward_post_process(args: Namespace, samples: list[Sample] | list[list[Sample]]):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    return raw_rewards, raw_rewards
