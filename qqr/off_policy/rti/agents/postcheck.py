import json
from .base import BaseAgent, AgentResult
from ..utils.text import strip_code_fence
from ..prompts import postcheck_judge as default_prompt_post


class PostcheckJudgeAgent(BaseAgent):
    name = "postcheck_judge"

    def __init__(self, client, enabled=True, prompt_module=None):
        super().__init__(client, enabled=enabled)
        self.prompt = prompt_module or default_prompt_post

    async def judge(self, *, user_prompt: str, history: str, original: str, corrected: str, tools: list, label: dict= None) -> AgentResult:
        prompt = self.prompt.build(user_prompt, history, original, corrected, tools, label)
        res = await self.run(prompt=prompt, meta={"task": "postcheck"})
        if not res.raw:
            res.parsed = {"ok": True, "diagnosis": "postcheck_judge_unavailable"}
            res.ok = True
            return res
        try:
            parse_content = self.parse_qwen3_output(res.raw)
            j = json.loads(strip_code_fence(parse_content['content']))
            res.parsed = j
            res.ok = True
        except Exception as e:
            res.parsed = {"ok": True, "diagnosis": "postcheck_parse_error", "error": str(e), "raw": res.raw}
            res.ok = False
        return res
