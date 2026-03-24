import asyncio
import logging
import re
from argparse import Namespace

from qqr.llms import OpenAI
from qqr.reward_models import get_reward_model
from qqr.schemas import LLMRewardModel, Sample

from . import config

logger = logging.getLogger(__name__)


class TravelLLMJudge(LLMRewardModel):
    def __init__(self):
        llm = OpenAI(
            model=config.llm_judge_model,
            api_key=config.llm_judge_api_key,
            base_url=config.llm_judge_base_url,
            concurrency_limit=config.llm_judge_concurrency_limit,
        )
        super().__init__(llm)

        self.system_prompt = config.llm_judge_system_prompt

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

    async def _compute_unidirectional(
        self, prediction: list[dict], reference: list[dict], query: str
    ) -> tuple[float, float]:
        trajectory_a, answer_a = self.process_messages(prediction)
        trajectory_b, answer_b = self.process_messages(reference)

        prompt = f"""<USER_QUERY>\n{query}\n</USER_QUERY>\n\n<PATH_A>\n{trajectory_a}\n</PATH_A>\n\n<PATH_B>\n{trajectory_b}\n</PATH_B>\n\n<Answer_A>\n{answer_a}\n</Answer_A>\n\n<Answer_B>\n{answer_b}\n</Answer_B>"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        score_a, score_b = 5.0, 5.0
        try:
            response = await self.llm(messages=messages, temperature=0.0)
            score_a, score_b = self.get_judge_scores(
                response.choices[0].message.content
            )

        except Exception as e:
            logger.warning(f"[LLMJudge] Failed to get result: {e}")

        return score_a, score_b

    async def compute(
        self, prediction: list[dict], reference: list[dict], query: str
    ) -> dict[str, float]:
        results = await asyncio.gather(
            self._compute_unidirectional(prediction, reference, query=query),
            self._compute_unidirectional(reference, prediction, query=query),
        )

        score_prediction = results[0][0] + results[1][1]
        score_reference = results[0][1] + results[1][0]

        return {"prediction": score_prediction, "reference": score_reference}

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

    result = await llm_judge(prediction, reference, query=query)
    pred_score, ref_score = result["prediction"], result["reference"]

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
