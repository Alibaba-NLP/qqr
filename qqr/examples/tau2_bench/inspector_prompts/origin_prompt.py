import json
from typing import Any, Dict, List


def build(history: str, original_response: dict, feedback_json: str, label: dict, tools: List, output_format: str = "json") -> str:
    """Build the original (baseline) agent prompt without any groundtruth or feedback.

    This prompt mirrors what the agent actually sees during rollout (rollout.py L492-494):
    a system policy message followed by the conversation history and available tools,
    with NO privileged information (no groundtruth, no evaluation criteria, no feedback).
    It serves as the counterfactual baseline to measure the probability shift introduced
    by the calibrated corrector prompt.
    """
    output_instruction = """
        Output JSON only, with exactly three fields:
        {{
            "reasoning": "Your reasoning about the current state and next best action.",
            "tool_calls": [
                {{
                    "name": "tool_name",
                    "arguments": {{...}}
                }}
            ],
            "content": "Your response content to the user."
        }}
        """

    if output_format == "tool_call_json":
        output_instruction = """
        Output in the following format:
        - Before the last dialogue turn, you must reason about the current user request and history, then output:
        <think>Your reasoning and plan for next action
        </think>
        <tool_call>{{"name": "tool_name1", "arguments": {{"arg1": "value1"}}}}</tool_call>
        <tool_call>{{"name": "tool_name2", "arguments": {{"arg1": "value1"}}}}</tool_call>
        """
    elif output_format == "tool_call_xml":
        output_instruction = """
        Output in the following format:
        - Before the last dialogue turn, you must reason about the current user request and history, then output:
        <think>Your reasoning and plan for next action
        </think>
        <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
        <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
        """

    return f"""
        You are a customer service agent. Based on the conversation history so far, generate the next action.

        [Conversation History]
        {history}

        [Available Tools]
        {json.dumps(tools, ensure_ascii=False, indent=4)}

        {output_instruction}
        """
