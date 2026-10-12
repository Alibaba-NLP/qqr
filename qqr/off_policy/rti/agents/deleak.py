from .base import BaseAgent, AgentResult
from ..prompts import deleak as default_prompt_deleak


class DeLeakAgent(BaseAgent):
    name = "de_leak"

    def __init__(self, client, enabled=True, prompt_module=None):
        super().__init__(client, enabled=enabled)
        self.prompt = prompt_module or default_prompt_deleak

    async def rewrite(self, *, history: str, draft_reasoning: str, draft_calls: str) -> AgentResult:
        prompt = self.prompt.build(history, draft_reasoning, draft_calls)
        res = await self.run(prompt=prompt, meta={"task": "deleak"})
        if not res.raw:
            res.raw = draft_reasoning or ""
        return res
