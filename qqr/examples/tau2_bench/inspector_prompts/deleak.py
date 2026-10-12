import json
from typing import List


def build(history, draft_reasoning, draft_calls) -> str:
    return f"""
        Rewrite a potentially leaked reasoning draft into reasoning that an agent could naturally produce WITHOUT knowing any reference/groundtruth or evaluation criteria.

        MUST satisfy:
        1) Use ONLY information explicitly present in the conversation history (including any tool outputs shown there). Do not introduce new facts, values, or conclusions not derivable from visible history.
        2) Remove any "I already know the answer / according to reference / evaluation requires / I should call X because it is required" vibe. No certainty without justification visible in history; use forward-looking language (e.g., "let me check / I need to verify").
        3) The rewritten reasoning must lead directly to the given tool call(s): state the minimal intent (what to query/check and why). Do NOT change, add, delete, or reorder tool calls.
        4) If the draft refers to any required action or criterion not visible in history, rephrase it as a natural inference from the customer's request and prior tool results.

        [Observable Conversation History]
        {history}

        [Potentially Leaky Reasoning Draft]
        {draft_reasoning}

        [Target Tool Calls (DO NOT MODIFY)]
        {draft_calls}
        """.strip()
