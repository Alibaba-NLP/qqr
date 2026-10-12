# reward_model.py
import logging
import re
import json
from argparse import Namespace
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict
import asyncio
from qqr.schemas import Sample

from . import config

logger = logging.getLogger(__name__)

# ---------------------------
# Navigation reward core utils
# ---------------------------
def _parse_nav_args(args: Dict) -> Tuple[Tuple[float, float], Tuple[float, float], List[Tuple[float, float]]]:
    s_lon = float(args["origin"].split(',')[0])
    s_lat = float(args["origin"].split(',')[1])
    e_lon = float(args["destination"].split(',')[0])
    e_lat = float(args["destination"].split(',')[1])
    via_raw = args.get("waypoints") or ""
    via_list: List[Tuple[float, float]] = []
    for seg in via_raw.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        lon_str, lat_str = seg.split(",")
        via_list.append((float(lon_str), float(lat_str)))
    return (s_lon, s_lat), (e_lon, e_lat), via_list


def _float_eq(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def _point_eq(p1: Tuple[float, float], p2: Tuple[float, float], eps: float = 1e-6) -> bool:
    return _float_eq(p1[0], p2[0], eps) and _float_eq(p1[1], p2[1], eps)


def _list_eq_as_set(l1: List[Tuple[float, float]], l2: List[Tuple[float, float]]) -> bool:
    if len(l1) != len(l2):
        return False
    used = [False] * len(l2)
    for p in l1:
        found = False
        for i, q in enumerate(l2):
            if not used[i] and _point_eq(p, q):
                used[i] = True
                found = True
                break
        if not found:
            return False
    return True

def navigation_reward(pred_nav: Dict, gold_nav: Dict, order: int) -> Tuple[int, str]:
    """
    pred_nav, gold_nav: {"name": "...", "arguments": {...}}
    order: 0~4
    """
    try:
        if pred_nav.get("name") != config.navigation_reward_name:
            return 0, f"预测工具调用 name 不是 {config.navigation_reward_name}"
        if gold_nav.get("name") != config.navigation_reward_name:
            return 0, f"gold 工具调用 name 不是 {config.navigation_reward_name}"

        gold_args = gold_nav.get("arguments") or {}
        pred_args = pred_nav.get("arguments") or {}

        gold_start, gold_end, gold_vias = _parse_nav_args(gold_args)
        pred_start, pred_end, pred_vias = _parse_nav_args(pred_args)

        if order == 0:
            gold_points = [gold_start] + gold_vias + [gold_end]
            pred_points = [pred_start] + pred_vias + [pred_end]
            if not _list_eq_as_set(gold_points, pred_points):
                return 0, f"order=0：所有点集合不匹配"
            return 1, "order=0：所有点集合匹配"

        if order == 1:
            if not _point_eq(pred_start, gold_start):
                return 0, f"order=1：起点不匹配"
            gold_rest = gold_vias + [gold_end]
            pred_rest = pred_vias + [pred_end]
            if not _list_eq_as_set(pred_rest, gold_rest):
                return 0, f"order=1：预测中间+终点集合未完全包含 gold"
            return 1, "order=1：起点匹配且中间+终点集合包含 gold"

        if order == 2:
            if not _point_eq(pred_end, gold_end):
                return 0, f"order=2：终点不匹配"
            gold_prefix = [gold_start] + gold_vias
            pred_prefix = [pred_start] + pred_vias
            if not _list_eq_as_set(pred_prefix, gold_prefix):
                return 0, f"order=2：预测起点+途经点集合未完全包含 gold"
            return 1, "order=2：终点匹配且起点+途经点集合包含 gold"

        if order == 3:
            if not _point_eq(pred_start, gold_start):
                return 0, f"order=3：起点不匹配"
            if not _point_eq(pred_end, gold_end):
                return 0, f"order=3：终点不匹配"
            if not _list_eq_as_set(pred_vias, gold_vias):
                return 0, f"order=3：预测途经点未完全包含 gold"
            return 1, "order=3：起点终点一致且途经点包含 gold"

        if order == 4:
            if not _point_eq(pred_start, gold_start):
                return 0, f"order=4：起点不匹配"
            if not _point_eq(pred_end, gold_end):
                return 0, f"order=4：终点不匹配"
            if len(pred_vias) != len(gold_vias):
                return 0, f"order=4：途经点数量不一致"
            for i, (g, p) in enumerate(zip(gold_vias, pred_vias)):
                if not _point_eq(g, p):
                    return 0, f"order=4：第{i}个途经点不一致"
            return 1, "order=4：起点终点一致且途经点顺序完全一致"

        return 0, f"未知的 order 值：{order}"

    except Exception as e:
        return 0, f"navigation_reward 执行异常: {repr(e)}"


# ---------------------------
# Format and repetition check for entire conversation
# ---------------------------
def _normalize_args(args: Any) -> Any:
    """
    规范化参数，用于比较
    """
    if isinstance(args, str):
        try:
            return json.loads(args)
        except:
            return args
    elif isinstance(args, dict):
        return {k: _normalize_args(v) for k, v in sorted(args.items())}
    elif isinstance(args, list):
        return [_normalize_args(v) for v in args]
    else:
        return args


def _args_to_str(args: Any, max_len: int = 1024) -> str:
    """
    将参数转换为可读的字符串表示（用于日志）
    """
    try:
        if isinstance(args, str):
            # 尝试解析JSON
            try:
                args_obj = json.loads(args)
                args_str = json.dumps(args_obj, ensure_ascii=False, sort_keys=True)
            except:
                args_str = args
        else:
            args_str = json.dumps(args, ensure_ascii=False, sort_keys=True)
        
        if len(args_str) > max_len:
            return args_str[:max_len] + "..."
        return args_str
    except:
        return str(args)


def _are_args_equal(args1: Any, args2: Any) -> bool:
    """
    比较两个参数是否相等（支持字典和JSON字符串）
    """
    # 规范化参数
    norm1 = _normalize_args(args1)
    norm2 = _normalize_args(args2)
    
    # 比较规范化后的结果
    return norm1 == norm2


def check_conversation_format(messages: list[dict]) -> Tuple[float, List[str]]:
    """
    检查整个对话的格式和重复调用：
    1. 所有中间步骤（有 tool_calls 的 assistant 消息）必须有 reasoning_content
    2. 所有 tool_calls 必须格式正确
    3. direction 只允许调用一次
    4. 其他函数可以多次调用，但参数必须不同
    """
    issues = []
    format_score = 1.0
    
    # 记录所有工具调用，用于检测重复
    nav_calls = []  # direction 调用
    other_calls = defaultdict(list)  # 其他函数的调用，按函数名分组
    
    # 遍历所有 assistant 消息
    assistant_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"]

    if len(assistant_messages) <= 1:
        issues.append(f"只进行了一轮对话")
        format_score *= 0

    for msg_idx, msg in enumerate(assistant_messages):
        tool_calls = msg.get("tool_calls")
        reasoning = msg.get("reasoning_content")

        if msg_idx == len(assistant_messages) - 1:
            if len(nav_calls) == 0 or len(other_calls) == 0:
                issues.append(f"所有轮次都没有调用合适的工具")
                format_score *= 0
            continue

        # 如果有 tool_calls，必须要有 reasoning_content
        if len(tool_calls) == 0:
            issues.append(f"第{msg_idx+1}条assistant消息没有调用tool_calls")
            format_score *= 0

        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            if not reasoning:
                issues.append(f"第{msg_idx+1}条assistant消息有tool_calls但缺少reasoning_content")
                format_score *= 0
            
            # 检查每个 tool_call
            for tc_idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    issues.append(f"第{msg_idx+1}条消息的第{tc_idx+1}个tool_call不是字典格式")
                    format_score *= 0.5
                    continue
                
                # 检查 tool_call 格式
                tc_score, tc_issues = _check_single_tool_call_format(tc, msg)
                if tc_score < 1.0:
                    for issue in tc_issues:
                        issues.append(f"第{msg_idx+1}条消息: {issue}")
                    format_score *= tc_score
                
                # 提取工具调用信息用于重复检测
                function = tc.get("function", {})
                func_name = function.get("name")
                if func_name:
                    args = function.get("arguments", {})
                    
                    # 获取参数的可读形式
                    args_str = _args_to_str(args)
                    
                    if func_name == config.navigation_reward_name:
                        # direction 调用
                        nav_calls.append({
                            "msg_idx": msg_idx,
                            "tc_idx": tc_idx,
                            "args": args,
                            "args_str": args_str
                        })
                    else:
                        # 其他函数调用
                        other_calls[func_name].append({
                            "msg_idx": msg_idx,
                            "tc_idx": tc_idx,
                            "args": args,
                            "args_str": args_str
                        })
    
    # # 检查 direction 重复调用
    # if len(nav_calls) > 1:
    #     # 生成详细的重复调用信息
    #     call_details = []
    #     for call in nav_calls:
    #         call_details.append(f"第{call['msg_idx']+1}条消息(参数: {call['args_str']})")
        
    #     issues.append(
    #         f"direction 被调用了 {len(nav_calls)} 次，但只允许调用1次: " + 
    #         " -> ".join(call_details)
    #     )
    #     format_score *= 0.3  # 严重惩罚
    
    # # 检查其他函数的重复调用（相同参数）
    # for func_name, calls in other_calls.items():
    #     if len(calls) <= 1:
    #         continue
        
    #     # 按参数分组
    #     args_groups = defaultdict(list)
    #     for call in calls:
    #         # 使用规范化后的参数作为key
    #         norm_args = _normalize_args(call["args"])
    #         args_key = json.dumps(norm_args, sort_keys=True)
    #         args_groups[args_key].append(call)
        
    #     # 检查每个参数组
    #     for args_key, group_calls in args_groups.items():
    #         if len(group_calls) > 1:
    #             # 获取参数的可读形式（使用第一个调用的args_str）
    #             sample_call = group_calls[0]
    #             args_str = sample_call["args_str"]
                
    #             # 生成调用位置列表
    #             positions = [f"第{call['msg_idx']+1}条消息" for call in group_calls]
                
    #             issues.append(
    #                 f"函数 {func_name} 有 {len(group_calls)} 次相同参数的调用: "
    #                 f"参数内容: {args_str}, 出现在: {', '.join(positions)}"
    #             )
    #             format_score *= 0.5
    
    # # 检查对话完成度
    # if assistant_messages:
    #     last_assistant = assistant_messages[-1]
    #     last_tool_calls = last_assistant.get("tool_calls")
        
    #     # 如果最后一条消息还有 tool_calls，说明对话可能未完成
    #     if last_tool_calls and isinstance(last_tool_calls, list) and len(last_tool_calls) > 0:
    #         # 检查是否包含 direction
    #         has_nav = False
    #         nav_args_str = ""
    #         for tc in last_tool_calls:
    #             if isinstance(tc, dict):
    #                 func_name = tc.get("function", {}).get("name")
    #                 if func_name == config.navigation_reward_name:
    #                     has_nav = True
    #                     args = tc.get("function", {}).get("arguments", {})
    #                     nav_args_str = _args_to_str(args)
    #                     break
            
    #         if has_nav:
    #             issues.append(
    #                 f"对话未完成：最后一条消息仍有 direction 调用 (参数: {nav_args_str})"
    #             )
    #             format_score *= 0.5
    
    return format_score, issues


def _check_single_tool_call_format(tc: dict, msg: dict) -> Tuple[float, List[str]]:
    """
    检查单个工具调用的格式是否正确
    返回：(分数, 问题列表)
    """
    issues = []
    score = 1.0
    
    # 检查必要字段
    if "id" not in tc:
        issues.append("tool_call 缺少 id 字段")
        score *= 0.8
    
    if tc.get("type") != "function":
        issues.append("tool_call type 不是 function")
        score *= 0.8
    
    function = tc.get("function")
    if not isinstance(function, dict):
        issues.append("tool_call function 字段格式错误")
        score *= 0.5
        return score, issues
    
    # 检查 function 的必要字段
    if "name" not in function:
        issues.append("tool_call 缺少 function.name")
        score *= 0.8
    
    if "arguments" not in function:
        issues.append("tool_call 缺少 function.arguments")
        score *= 0.8
    else:
        # 检查 arguments 是否为有效的 JSON 字符串或字典
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                json.loads(args)
            except json.JSONDecodeError:
                issues.append("tool_call arguments 不是有效的 JSON 字符串")
                score *= 0.5
        elif not isinstance(args, dict):
            issues.append("tool_call arguments 必须是字典或 JSON 字符串")
            score *= 0. 
    
    # 检查推理过程
    if len(msg["reasoning_content"]) == 0:
        issues.append("缺少推理过程")
        score *= 0.1
    
    return max(0.0, score), issues


# ---------------------------
# Combined reward function
# ---------------------------
async def eval_reward(args: Namespace, sample: Sample, **kwargs):
    """
    综合评估奖励：
    1. 导航任务奖励（判断最终的 navigation 调用是否正确）
    2. 格式奖励（检查整个对话的格式、重复调用等）
    """
    total_reward = 0.0
    reasons = []
    
    # 1. 格式奖励 - 检查整个对话
    format_score, format_issues = check_conversation_format(sample.messages)
    # total_reward += format_score * float(config.format_reward_weight)
    if format_issues:
        # 只记录前3个问题，避免日志过长
        top_issues = format_issues
        reasons.append(f"格式不规范直接判错{'; '.join(top_issues)}")
    else:
        reasons.append(f"格式奖励({format_score:.2f}): 格式正确")
    
        # 2. 导航任务奖励（如果启用了导航模式）
        # 注意：label 坐标更新已在 rollout.py agent_loop 开始前完成，此处无需重复调用
        if config.reward_mode == "pointwise_navigation":

            # 获取最后一次正确的 navigation 调用
            pred_nav = _extract_last_tool_call_from_messages(sample.messages, config.navigation_reward_name)
            
            if pred_nav is None:
                reasons.append(f"未找到正确的 navigation 工具调用")
            else:
                gold_nav = _extract_groundtruth_from_label(sample.label)
                if gold_nav is not None:
                    order = _parse_order_from_label(sample.label)
                    nav_r, nav_reason = navigation_reward(pred_nav=pred_nav, gold_nav=gold_nav, order=order)
                    total_reward = nav_r
                    reasons.append(f"导航奖励({nav_r}): {nav_reason}")
    
    sample.reward = total_reward
    if total_reward == 1:
        print()
    if total_reward == 0:
        print()
    logger.debug(f"[reward] total={total_reward:.2f}, details: {'; '.join(reasons)}")


def _extract_direction_calls_from_messages(messages: list[dict]) -> List[Dict[str, Any]]:
    """
    从消息中提取所有 direction 工具调用，并尝试合并连续的调用
    
    返回合并后的 direction 调用列表，每个元素格式为：
    {"name": "direction", "arguments": {...}}
    """
    direction_calls = []
    
    # 首先收集所有 direction 调用
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function", {})
            if function.get("name") != config.navigation_reward_name:
                continue

            args = function.get("arguments") or {}
            # 有些实现 arguments 可能是 JSON 字符串；尽量兼容
            if isinstance(args, str):
                try:
                    import json
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            direction_calls.append({"name": config.navigation_reward_name, "arguments": args})
    
    # 尝试合并连续的 direction 调用
    if len(direction_calls) <= 1:
        return direction_calls
    
    merged_calls = []
    i = 0
    
    while i < len(direction_calls):
        current_call = direction_calls[i]
        current_args = current_call["arguments"]
        
        # 检查是否可以与后续调用合并
        j = i + 1
        waypoints = []
        
        # 收集当前调用的已有 waypoints
        current_waypoints = current_args.get("waypoints", "")
        if current_waypoints:
            waypoints.extend([wp.strip() for wp in current_waypoints.split(";") if wp.strip()])
        
        # 尝试合并后续连续的调用
        while j < len(direction_calls):
            next_call = direction_calls[j]
            next_args = next_call["arguments"]
            
            # 检查是否可以合并：当前调用的终点 == 下一个调用的起点
            current_dest = current_args.get("destination", "")
            next_origin = next_args.get("origin", "")
            
            if current_dest and next_origin and current_dest == next_origin:
                # 可以合并
                # 将下一个调用的起点添加到 waypoints（这是关键的途经点）
                waypoints.append(next_origin)
                
                # 添加下一个调用的 waypoints
                next_waypoints = next_args.get("waypoints", "")
                if next_waypoints:
                    waypoints.extend([wp.strip() for wp in next_waypoints.split(";") if wp.strip()])
                
                # 更新当前调用的终点为下一个调用的终点
                current_args["destination"] = next_args.get("destination", "")
                j += 1
            else:
                # 不能合并，跳出循环
                break
        
        # 构造合并后的调用
        merged_args = current_args.copy()
        if waypoints:
            merged_args["waypoints"] = ";".join(waypoints)
        
        merged_calls.append({"name": config.navigation_reward_name, "arguments": merged_args})
        i = j  # 移动到下一个未处理的调用
    
    return merged_calls


# def _extract_last_tool_call_from_messages(messages: list[dict], tool_name: str) -> Optional[Dict[str, Any]]:
#     """
#     从消息中提取最后一次正确的工具调用（已修改为支持合并 direction 调用）
#     """
#     if tool_name == config.navigation_reward_name:
#         # 对于 direction 工具，使用新的合并逻辑
#         merged_calls = _extract_direction_calls_from_messages(messages)
#         return merged_calls[-1] if merged_calls else None
#     else:
#         # 对于其他工具，保持原有逻辑
#         last = None
#         for msg in messages:
#             if not isinstance(msg, dict):
#                 continue
#             if msg.get("role") != "assistant":
#                 continue

#             tool_calls = msg.get("tool_calls") or []
#             if not isinstance(tool_calls, list):
#                 continue

#             for tc in tool_calls:
#                 if not isinstance(tc, dict):
#                     continue
#                 function = tc.get("function", {})
#                 if function.get("name") != tool_name:
#                     continue

#                 args = function.get("arguments") or {}
#                 # 有些实现 arguments 可能是 JSON 字符串；尽量兼容
#                 if isinstance(args, str):
#                     try:
#                         import json
#                         args = json.loads(args)
#                     except Exception:
#                         args = {}
#                 if not isinstance(args, dict):
#                     args = {}

#                 last = {"name": tool_name, "arguments": args}
#         return last



def _extract_last_tool_call_from_messages(messages: list[dict], tool_name: str) -> Optional[Dict[str, Any]]:
    """
    从消息中提取最后一次正确的工具调用
    """
    last = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function", {})
            if function.get("name") != tool_name:
                continue

            args = function.get("arguments") or {}
            # 有些实现 arguments 可能是 JSON 字符串；尽量兼容
            if isinstance(args, str):
                try:
                    import json
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            last = {"name": tool_name, "arguments": args}
    return last



def _extract_groundtruth_from_label(label: Any) -> Optional[Dict[str, Any]]:
    """
    适配你给的 reference 示例：
      sample.label = {"reference": {"groundtruth": {...}, "order": "...", ...}}
    """
    if not isinstance(label, dict):
        return None
    ref = label.get("reference")
    if isinstance(ref, dict) and isinstance(ref.get("groundtruth"), dict):
        return ref["groundtruth"]
    if isinstance(label.get("groundtruth"), dict):
        return label["groundtruth"]
    return None


def _parse_order_from_label(label: Any) -> int:
    """
    从 "1.用户明确了起点..." 解析 1；失败用默认。
    """
    default = int(config.navigation_reward_order_default)
    if not isinstance(label, dict):
        return default
    ref = label.get("reference")
    order_text = None
    if isinstance(ref, dict):
        order_text = ref.get("order")
    if order_text is None:
        order_text = label.get("order")
    if not isinstance(order_text, str):
        return default
    m = re.match(r"\s*(\d+)\s*[\.、]", order_text)
    if not m:
        return default
    try:
        v = int(m.group(1))
        return v
    except Exception:
        return default


async def group_reward(args: Namespace, group: list[list[Sample]], **kwargs):
    """
    仍保留接口：group 是 list[list[Sample]]（每条轨迹一个 list）。
    """
    if len(group) <= 0:
        raise ValueError("group must be non-empty")

    for traj in group:
        if len(traj) == 0:
            continue
        last = traj[-1]
        await eval_reward(args, last)

        # 回填：让同一 rollout 里每个 step 的 sample.reward 一致（常见做法）
        for s in traj:
            s.reward = last.reward


def reward_post_process(args: Namespace, samples: list[Sample] | list[list[Sample]]):
    """
    保持你原来的返回格式：raw_rewards, raw_rewards
    """
    if not samples:
        return [], []

    if isinstance(samples[0], list):
        flat = [s for traj in samples for s in traj]
        raw_rewards = [s.get_reward_value(args) for s in flat]
        return raw_rewards, raw_rewards

    raw_rewards = [s.get_reward_value(args) for s in samples]
    return raw_rewards, raw_rewards




