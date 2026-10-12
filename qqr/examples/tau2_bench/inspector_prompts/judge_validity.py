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


def build(history_json: str, response_text: str, label: dict) -> str:
    domain = label.get("domain") or ""
    evaluation_criteria = label.get("evaluation_criteria") or {}
    actions = evaluation_criteria.get("actions") or []
    nl_assertions = evaluation_criteria.get("nl_assertions") or []
    communicate_info = evaluation_criteria.get("communicate_info") or []
    env_assertions = evaluation_criteria.get("env_assertions") or []

    # ── action partitioning ─────────────────────────────────────────────────
    # telecom: agent-side vs user-side (device); retail/airline: all agent-side
    if domain == "telecom":
        agent_actions = [a for a in actions if a.get("requestor") == "assistant"]
        user_actions  = [a for a in actions if a.get("requestor") == "user"]
    else:
        agent_actions = actions
        user_actions  = []

    # ── domain-specific blocks ──────────────────────────────────────────────

    # Rule 4 — USER_SIDE_DEVICE_ACTION (telecom only)
    if domain == "telecom":
        rule4 = f"""4) USER_SIDE_DEVICE_ACTION: The agent emits a tool call for a user-device action.
           The agent must NEVER call these — it must instruct the user to perform them on their device.
           USER_SIDE_TOOLS  = {json.dumps(sorted(USER_SIDE_TOOLS), ensure_ascii=False)}
           AGENT_SIDE_TOOLS = {json.dumps(sorted(AGENT_SIDE_TOOLS), ensure_ascii=False)}"""
        rule_offset = 1  # rules 5-7 come after rule 4
    elif domain == 'retail':
        rule4 = '4) MISSING_CONFIRMATION: Before any tool call that updates the database (including modify_user_address and any cancel/modify/return/exchange action), the agent must first list the exact action details and obtain an explicit user confirmation "yes"; if it updates first and asks for confirmation afterward, mark invalid.'
        rule_offset = 1  # rules 4-6 directly follow rule 3
    else:
        rule4 = '4）MISSING_CONFIRMATION: Before taking any actions that update the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), the agent must first list the action details and obtain an explicit user confirmation "yes"; if it updates first and asks for confirmation afterward, mark invalid.'
        rule_offset = 1  # rules 4-6 directly follow rule 3

    n4, n5, n6, n7 = 4 + rule_offset, 5 + rule_offset, 6 + rule_offset, 7 + rule_offset

    # Completion check description
    if domain == "telecom":
        done_check = (
            "- agent-side action: a matching assistant tool call + successful result in history.\n"
            "           - user-side action: a matching user tool call output in history (user executed it on device)."
        )
        user_actions_block = f"""
           Required user-side actions (agent must GUIDE the user to perform on their device):
           {user_actions}"""
        completion_note = (
            "all required actions executed AND the final environment state satisfies env_assertions"
        )
        env_block = f"""
        [Completion reference — telecom]
        env_assertions: {json.dumps(env_assertions, ensure_ascii=False)}"""
        suggest_hint = (
            "Specify the exact tool the agent should call, OR the device action the agent should"
            " instruct the user to perform."
        )
        error_types = (
            '"fabricated_arguments" | "pointless_repeat" | "false_claim" | "user_side_device_action"'
            ' | "premature_termination" | "contradictory_action" | "goal_drift"'
        )
    else:
        done_check = "- agent-side action: a matching assistant tool call + successful result in history."
        user_actions_block = ""
        completion_note = "all required actions executed successfully"
        env_block = ""
        suggest_hint = "Specify the exact tool the agent should call next and with what arguments."
        if domain == "retail":
            error_types = (
                '"fabricated_arguments" | "pointless_repeat" | "false_claim" | "missing_confirmation" '
                ' | "premature_termination" | "contradictory_action" | "goal_drift"'
            )
        else:
            error_types = (
                '"fabricated_arguments" | "pointless_repeat" | "false_claim" | "missing_confirmation" '
                ' | "premature_termination" | "contradictory_action" | "goal_drift"'
            )

    judge_prompts = f"""
        You are a trajectory supervisor for a tool-calling agent task. Your job is to judge whether the
        CURRENT response is a valid next step that keeps the agent on track toward completing the task.
        Task is complete only when {completion_note}.

        [Observable History]
        {history_json}

        [Current Response](Formatted as json)
        {response_text}

        [Required actions to complete the task]
        Agent-side actions (the agent must call these tools itself):
        {agent_actions}
        {user_actions_block}
        How to tell if an action is done:
        {done_check}

        RULES — mark invalid if ANY rule is violated:
        1) FABRICATED_ARGUMENTS: A tool argument value is not present in history and not derivable from the task description. (Values from earlier tool outputs ARE valid.)
        2) POINTLESS_REPEAT: Calling the exact same tool with the exact same arguments that already succeeded. (Same tool, different arguments = allowed.)
        3) FALSE_CLAIM: The response claims information was retrieved without a corresponding tool call. 
        {rule4}
        {n4}) PREMATURE_TERMINATION: The response ends the conversation (transfer to human, closing message) while one or more required actions are not yet done.
        {n5}) CONTRADICTORY_ACTION: The response calls a tool or instructs the user in a way that directly contradicts a required action (e.g., cancels what should be exchanged, disables what must be on).
        {n6}) GOAL_DRIFT: The response ignores an obviously pending required action and pursues something unrelated, making it harder to get back on track.
        {n7}) ARGUMENT_MISMATCH: For actions that update the database (e.g., update/change/book/cancel/modify), if the [Required actions] list specifies exact argument values (such as specific baggage counts, payment splits, or passenger details), the agent's tool call must match these values exactly. Deviating from the required parameter values, even if logically justified by the agent, is considered a failure to complete the required action correctly.

        DO NOT mark invalid for:
        - Suboptimal but recoverable action order
        - Diagnostic / information-lookup steps before acting
        - Re-attempting a failed action with corrected arguments
        - A valid path that differs from the reference
        {env_block}

        [Evaluation_Criteria — for context only]
        {evaluation_criteria}

        Output JSON only:
        {{
        "diagnosis": "If invalid: state which rule was violated and quote/describe the offending part. If valid: briefly confirm the step is on track.",
        "is_valid": true/false,
        "error_types": [{error_types}],
        "suggest_next": "{suggest_hint} Null if valid."
        }}
        """.strip()

    return judge_prompts