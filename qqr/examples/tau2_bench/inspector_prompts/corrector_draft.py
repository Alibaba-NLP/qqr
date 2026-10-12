import json
from typing import Any, Dict, List

USER_SIDE_TOOLS = {
    "check_status_bar", "check_network_status", "check_network_mode_preference",
    "check_sim_status", "check_data_restriction_status", "check_apn_settings",
    "check_wifi_status", "check_wifi_calling_status", "check_vpn_status",
    "check_installed_apps", "check_app_status", "check_app_permissions",
    "run_speed_test", "can_send_mms",
    "set_network_mode_preference", "toggle_airplane_mode", "reseat_sim_card",
    "toggle_data", "toggle_roaming", "toggle_data_saver_mode", "set_apn_settings",
    "reset_apn_settings", "toggle_wifi", "toggle_wifi_calling", "connect_vpn",
    "disconnect_vpn", "grant_app_permission", "reboot_device"
}

AGENT_SIDE_TOOLS=['get_customer_by_phone', 'get_customer_by_id', 'get_customer_by_name', 'get_details_by_id', 'suspend_line', 'resume_line', 'get_bills_for_customer', 'send_payment_request', 'get_data_usage', 'enable_roaming', 'disable_roaming', 'transfer_to_human_agents', 'refuel_data']


def build(history: str, original_response: dict, feedback_json: str, label: dict, tools: List, output_format: str = "json") -> str:
    domain = label.get("domain") or ""
    evaluation_criteria = label.get("evaluation_criteria") or {}
    actions = evaluation_criteria.get("actions") or []
    nl_assertions = evaluation_criteria.get("nl_assertions") or []
    communicate_info = evaluation_criteria.get("communicate_info") or []

    user_side_tools_block = ""
    user_side_node = ""
    if domain == "telecom":
        user_side_tools_block = f"""
        IMPORTANT: Follow the interaction model in the system prompt. If an action must be done by the user on their device, do not emit a tool call for it; guide the user to perform the action on their device instead.
        - USER_SIDE_DEVICE_ACTIONS = {json.dumps(sorted(USER_SIDE_TOOLS), ensure_ascii=False)}
        - AGENT_SIDE_TOOLS = {json.dumps(sorted(AGENT_SIDE_TOOLS), ensure_ascii=False)}
        """
        user_side_node = """
        Note: "tool_calls" must be an empty list if no agent-side tool is to be called. Never include user-side tools in tool_calls
        """
    elif domain=='retail':
        user_side_tools_block = f"""
        IMPORTANT: Follow the interaction model in the system prompt. Before any tool call that updates the database (including modify_user_address and any cancel/modify/return/exchange action), the agent must first list the exact action details and obtain an explicit user confirmation "yes".
        """
    else:
        user_side_tools_block = f"""
        IMPORTANT: Follow the interaction model in the system prompt. Before any tool call that updates the database (including send_certificate and any cancel/update action), the agent must first list the exact action details and obtain an explicit user confirmation "yes".
        """

    output_instruction = """
        Output JSON only, using this unified schema:

        {
            "reasoning": "Optimized reasoning text (style-aligned with prior draft). If keeping original, explain why it's acceptable.",
            "tool_calls": [
                {
                    "name": "tool_name",
                    "arguments": {...}
                }
            ],
            "content": "Optimized content."
        }

        Field rules:
        - "reasoning" is always required.
        - "tool_calls" and "content" are mutually exclusive in substance: use only one of them for the actual next action.
        - If an agent-side tool call should be made now:
            - set "tool_calls" to a non-empty list
            - set "content" to ""
        - If the assistant should reply directly to the user:
            - set "content" to a non-empty string
            - set "tool_calls" to be an empty list
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
        <tool_call><function=name>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>
        <tool_call><function=name>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>
        """

    return f"""
        You are a silent calibrator for a tool-calling agent task. Your role is to minimally correct the agent's next action to bring it closer to the successful path, WITHOUT exposing any privileged knowledge.

        [Conversation History]
        {history}

        [Available Tools]
        {json.dumps(tools, ensure_ascii=False, indent=4)}

        [Reference / evaluation criteria (for calibration only; MUST NOT appear in output)]
        {evaluation_criteria}

        CORE INSTRUCTIONS:
        Generate the optimized next action based on [Conversation History], NOT by continuing from [Prior Draft].
        The Prior Draft is provided only as a style reference.

        GENERATION STRATEGY:
        1. Style alignment: match the reasoning style, verbosity, and expression habits of the Prior Draft.
        2. Progressive correction: guide toward a more reasonable path; corrections should feel like natural evolution of thought.
        3. Concealment: NEVER mention "reference", "groundtruth", "required actions", "evaluation criteria", "prior draft", "calibrated response", "previous version", "I previously" or "the correct answer". All reasoning must appear naturally derived from conversation history.
        4. Minimal change: only correct clearly wrong steps; preserve correct parts of the prior draft.

        CORRECTION GUIDELINES:
        - If a tool argument is wrong, use the correct value from history (e.g., correct payment_id from user profile)
        - If the agent skipped a necessary step, add a message to gather required info or confirm with user
        - If the agent made an unfounded claim, revise to acknowledge uncertainty or check first
        - If multiple valid approaches exist, prefer the one closer to the original intent
        - DO NOT introduce information that cannot be derived from visible history
        - If no valid agent-side tool call is appropriate, set "tool_calls" to an empty list.

        {user_side_tools_block}

        [Prior Draft (style reference only)]
        {json.dumps(original_response, ensure_ascii=False, indent=4)}

        [Supervisor Feedback (problem indication)]
        {feedback_json}

        {output_instruction}

        {user_side_node}
        """
