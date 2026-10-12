def build(user_prompt, history, original, corrected, tools, label: dict = None) -> str:
    domain = label.get("domain") or ""
    evaluation_criteria = label.get("evaluation_criteria") or {}
    return f"""
        You are a post-calibration safety & faithfulness inspector. Decide whether the CALIBRATED reply should be accepted.

        VISIBLE INFO = user question + tool list + conversation history (including tool outputs). Nothing else is allowed.

        [User Question]
        {user_prompt}

        [Evaluation Criteria]
        {evaluation_criteria}

        [Available Tools]
        {tools}

        [Agent History (visible)]
        {history}

        [Prior Draft](Formatted as json)
        {original}

        [Calibrated Reply](Formatted as json)
        {corrected}

        Reject (ok=false) ONLY if ANY holds:
        1) leak: explicitly mentions "reference", "groundtruth", "evaluation criteria", "required actions", "answer key", "labels", "seeing the solution", or similar privileged information keywords.
        2) fabricated_tool_result: claims a specific tool result/value that does NOT appear in history. Example: claiming "flight price is $500" when no tool returned that exact value.
        3) wrong_tool_arguments: uses tool arguments that contradict visible history or deviate from explicit constraints defined in the [Evaluation Criteria]. Mandatory Alignment: For any tool call, if the [Evaluation Criteria] explicitly specifies a parameter value, the agent's argument must match it exactly. Contextual Discretion: If a parameter is not specified in the [Evaluation Criteria], the agent is permitted to derive the value logically based on the conversation history and applicable policies. Invalidity arises only when an explicitly required value is ignored or altered.
        4) unreasonable_calibration: the calibrated reply is not a reasonable next step given the visible history, fails to improve the prior draft", introduces unnecessary or less appropriate actions.

        ALLOWED (not an issue) - DO NOT REJECT for these:
        - Adding reasonable next steps that are implied by the task context (e.g., asking for user confirmation before database updates)
        - Adding policy-compliant behaviors (e.g., "I need to verify with the user first")
        - Changing the action sequence to be more cautious or thorough
        - Adding intermediate steps that logically follow from the conversation

        The calibration goal is to IMPROVE the agent's behavior, not just paraphrase it. Adding missing confirmation steps or policy-compliant actions is the INTENDED behavior of calibration.

        Output JSON only in the following format:
        {{
        "ok": true/false,
        "reasons": ["leak", "fabricated_tool_result", "wrong_tool_arguments", "unreasonable_calibration"],
        "diagnosis": "If ok is false, describe the exact problem and support the judgment with observable evidence from the conversation history, including specific quoted fragments and/or explicitly missing required points. If ok is true, briefly explain why the response is acceptable."
        }}

        **Most Important!: Reject leak: explicitly mentions privileged information regarding the evaluation process or the existence of alternative drafts. This includes keywords such as "reference", "groundtruth", "evaluation criteria", "required actions", "answer key", "labels", "seeing the solution", "The prior draft", "previous reply", "previous version", "calibrated version", or any phrasing that implies the model is comparing its current output to a hidden "prior draft" or "ground truth" text within the <think> block or the final response.**

        """.strip()
