import json
from .base import BaseAgent, AgentResult
from ..utils.text import strip_code_fence
from ..prompts import corrector_draft as default_prompt_correct
from ..prompts import corrector_draft_qwen8b as default_prompt_qwen8b_module
from ..prompts import origin_prompt as default_origin_prompt


class CorrectorDraftAgent(BaseAgent):
    name = "corrector_draft"

    def __init__(self, client, enabled=True, prompt_module=None, prompt_qwen8b_module=None, origin_prompt=None):
        super().__init__(client, enabled=enabled)
        self.prompt = prompt_module or default_prompt_correct
        self.origin_prompt = origin_prompt or default_origin_prompt

    async def draft(self, *, history: str, original_response: dict, feedback_json: str, label: dict, tools: list) -> AgentResult:
        prompt = self.prompt.build(history, original_response, feedback_json, label, tools)
        res = await self.run(prompt=prompt, meta={"task": "corrector_draft"}, return_message=True)
        if not res.raw:
            res.ok = False
            res.parsed = None
            return res
        try:
            parse_content = self.parse_qwen3_output(res.raw['content'])
            res.parsed = json.loads(strip_code_fence(parse_content['content']))
            res.ok = True
        except Exception as e:
            res.ok = False
            res.parsed = {"error": str(e), "raw": res.raw}
        return res
