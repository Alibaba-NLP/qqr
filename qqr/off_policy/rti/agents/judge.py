import json
from .base import BaseAgent, AgentResult
from ..utils.text import strip_code_fence
from ..prompts import judge_validity as default_prompt_judge  # fallback


class ValidityJudgeAgent(BaseAgent):
    name = "judge_validity"

    def __init__(self, client, enabled=True, prompt_module=None):
        super().__init__(client, enabled=enabled)
        self.prompt = prompt_module or default_prompt_judge

    async def judge(self, *, history_json: str, response_text: str, label: dict) -> AgentResult:
        prompt = self.prompt.build(history_json, response_text, label)
        res = await self.run(prompt=prompt, meta={"task": "judge_validity"})
        if not res.raw:
            res.parsed = {"is_valid": True, "diagnosis": "judge_unavailable"}
            res.ok = True
            return res
        try:
            parse_content = self.parse_qwen3_output(res.raw)
            j = json.loads(strip_code_fence(parse_content['content']))
            res.parsed = j
            res.ok = True
            return res
        except Exception as e:
            res.parsed = {"is_valid": True, "diagnosis": "judge_parse_error", "error": str(e), "raw": res.raw}
            res.ok = False
            return res
