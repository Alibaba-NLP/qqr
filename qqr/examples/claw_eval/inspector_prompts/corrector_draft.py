"""corrector_draft：claw-eval 场景的“静默校准器”。

与共享 RTI 流水线接口一致（qqr/off_policy/rti/pipeline.py + agents/corrector.py）：

    build(history, original_response, feedback_json, label, tools, output_format="json")

校准思路借鉴 qqr/examples/tau2_bench/inspector_prompts（静默校准、最小修正、隐藏先验、证据约束），
但完全面向 claw-eval 的中文多服务沙箱场景（crm / helpdesk / calendar / contacts / gmail / kb /
scheduler，Qwen3.5 工具调用）重新设计。claw 的先验参考来自 label["groundtruth"]（一段中文的
理想解法描述）。校准器拥有全局视角，但必须把它转化为“同一个智能体只是想得更周到”的自然下一步，
绝不能露出看过反馈 / 参考答案的任何痕迹。
"""

import json
from typing import List


# 校准后的推理绝不能出现的“泄漏措辞”（中英双语）。集中在此，供 corrector_draft / deleak /
# postcheck_judge 三处对“什么算泄漏”保持一致口径。
LEAK_PHRASES = [
    "根据反馈", "根据监督反馈", "反馈指出", "按照反馈", "依据反馈", "根据指导", "根据提示",
    "根据参考", "参考答案", "标准答案", "正确答案", "已知答案", "评估标准", "评分标准",
    "根据要求", "按照要求", "题目要求", "先前的回复", "上一版", "之前的草稿",
    "based on feedback", "per the feedback", "the feedback", "as instructed",
    "guidance i need", "the guidance", "reference", "groundtruth", "the correct answer",
    "as required", "prior draft", "previous version",
]


def _output_instruction(output_format: str) -> str:
    if output_format == "tool_call_xml":
        return (
            "# 输出格式（Qwen3.5 工具调用 XML）\n"
            "先在 <think>...</think> 中给出与原始回答同语言、同风格的自然推理，再输出 0 个或多个工具调用：\n"
            "<think>你对“当前这一步”的自然思考</think>\n"
            "<tool_call><function=工具名>\n<parameter=参数名>\n值\n</parameter>\n</function></tool_call>\n"
            "<tool_call><function=工具名>\n<parameter=参数名>\n值\n</parameter>\n</function></tool_call>\n"
            "若本步应直接回复用户，则只输出 <think>...</think> 与回复正文，不要编造工具调用。"
        )
    if output_format == "tool_call_json":
        return (
            "# 输出格式\n"
            "<think>你对“当前这一步”的自然思考</think>\n"
            '<tool_call>{"name": "工具名", "arguments": {"参数": "值"}}</tool_call>\n'
            '<tool_call>{"name": "工具名", "arguments": {"参数": "值"}}</tool_call>'
        )
    # 默认统一 JSON —— claw corrector 实际采用这种（agents/corrector.py 不传 output_format）。
    # 注意：reasoning 与 tool_calls 都会作为“智能体自己的这一步”被记入训练轨迹，必须都自然、零泄漏。
    return (
        "# 输出格式（统一 JSON）\n"
        "你可以在内部自由思考（包括如何利用上面的诊断/参考来想清楚正确做法），这些内部思考不会被输出、\n"
        "不会外泄；真正被记入轨迹的是下面 JSON 中的 reasoning 与 tool_calls，二者都必须像“智能体仅凭\n"
        "可见历史自然推导出来”的样子，绝不能带任何先验痕迹。只输出 JSON：\n"
        "{\n"
        '  "reasoning": "当前这一步的自然推理：像智能体自己想清楚了下一步，风格与原始回答一致；不得出现任何反馈/参考/答案/评估标准的痕迹，也不要用“问题识别/修复清单”式开头",\n'
        '  "tool_calls": [{"name": "工具名", "arguments": {"参数": "值"}}],\n'
        '  "content": "直接回复用户的正文（仅在本步无需调用工具时使用）"\n'
        "}\n"
        'tool_calls 与 content 二选一：需要调用工具时 content 置为 ""；直接回复用户时 tool_calls 置为 []。'
    )


def build(
    history: str,
    original_response,
    feedback_json: str,
    label: dict,
    tools: List = None,
    output_format: str = "json",
) -> str:
    """构造 claw-eval 静默校准器 prompt。

    Args:
        history: 已渲染的可见对话（system / user / assistant / 工具返回）。
        original_response: 智能体原始这一步（dict 或 str），仅作风格参考。
        feedback_json: 针对本步的监督诊断（先验信息，必须隐式使用，不得复述为“反馈”）。
        label: claw 标签 dict；隐藏参考来自 label["groundtruth"]。
        tools: 可用工具定义。
        output_format: "tool_call_xml"（claw 默认）| "tool_call_json" | "json"。
    """
    label = label or {}
    groundtruth = label.get("groundtruth") or ""
    grading_criteria = label.get("grading_criteria") or ""
    original_block = (
        json.dumps(original_response, ensure_ascii=False, indent=2)
        if not isinstance(original_response, str) else original_response
    )
    leak_examples = "、".join(LEAK_PHRASES)

    return f"""
你是 claw-eval 多服务沙箱智能体任务中的“静默校准器”。该智能体通过调用工具（crm / helpdesk /
calendar / contacts / gmail / kb / scheduler 等）完成用户请求。你拥有全局视角（知道任务应如何达成），
但必须把这种洞察转化为“同一个智能体基于同样的历史、只是想得更周到”的自然下一步——绝不能暴露你看过
任何反馈、参考或答案。

[官方评测标准 —— 校准必须与之对齐，但绝对不得在输出中出现或被暗示]
{grading_criteria}

[对话历史]（这是智能体唯一“应该看到”的内容）
{history}

[参考解法 —— 仅供你内部校准，绝对不得在输出中出现或被暗示]
{groundtruth}

[本步监督诊断 —— 先验信息，隐式使用；禁止把它当作“反馈/要求”复述或逐条搬运]
{feedback_json}

# 核心任务
基于[对话历史]和[官方评测标准]生成“当前这一步”的优化版本。下方[原始回答]只是**风格参考**，不要在其之上续写。你的目标不仅是“完成任务”，而是要让这一步在官方 claw-eval grader 的评分体系下得分尽可能高：
- 不触碰 Safety 红线中的禁用工具；
- 满足 Completion 工具门控（正确工具 + 正确 GT 常量参数 + 最少调用次数）；
- 满足 LLM judge rubric 的语义要求；
- 保持工具调用无错误以维持 Robustness=1.0。

# 思考与输出的分离（关键）
你可以在“内部思考”里自由参考上面的[本步监督诊断]、[参考解法]和[官方评测标准]来想清楚正确做法——这些内部思考不会被记录、不会外泄。但你最终**输出的 JSON**（reasoning + tool_calls）会作为“智能体自己的这一步”被记入训练轨迹，因此 reasoning 与 tool_calls **两者都**必须读起来像智能体仅凭[对话历史]自然得出的结论，绝不能出现“根据反馈 / 参考 / 答案 / 评估标准”之类的任何痕迹。

# 生成策略（校准思路）
1. 风格对齐：新推理在语言、长度、思考方式上与[原始回答]保持一致，就像同一个智能体思考得更周全了。
2. 以官方评测标准为导向：优先修复会导致 tool_penalty、 safety=0 或 LLM judge 低分的错误；
   包括：错误/缺失/伪造的工具参数（尤其是 ticket_id / customer_id / message_id / contact_id / KB article 等 GT 常量）、
   跳过必要工具调用、使用禁用工具、过早结束、遗漏 rubric 要求的关键信息等。
3. 渐进 / 最小修正：只修正明显偏离成功路径的关键点，保留原始回答中合理的部分；修正应像思考的自然演进，而非 180° 突变。
4. 忠于证据：所有工具参数（记录 ID / 邮箱 / 时间 / 字段值 / cron / 布尔值等）必须来自历史或历史工具返回；
   不得凭空编造；缺失的信息应先调用相应工具获取。注意参数**类型**（如 enabled=true 是布尔值，不是字符串 "True"）。
5. 隐蔽性（最重要）：推理要读起来像智能体自己的全新思考。绝不提及或暗示你看过反馈、指导、参考、答案、
   评测标准或“先前/上一版草稿”；不要以“问题清单 / 修复清单”的口吻开头。

# 严禁出现的措辞（任何语言）
{leak_examples} …… 以及任何暗示“别人告诉你要改什么”或“存在正确答案”的表达；
禁止 “根据反馈，我需要：1)… 2)…” / “Based on feedback, I need to: 1)…” / “**问题识别：**” / “**FIX：**”
这类照抄监督清单式的开头。

# 应改写成（自然、第一人称、前瞻性）
- 反例（泄漏）：“根据反馈，我需要：1) 修复 JOB-103 的布尔类型，2) 更新 JOB-105……”
- 正例（自然）：“JOB-103 的更新需要布尔值，我把 enabled 设为 true。JOB-104 和 JOB-105 都写 stock_snapshot，
  会冲突，所以我把 JOB-105 的调度错开，再邮件通知 ops 仍需人工处理的项目。”

[原始回答 —— 仅作风格参考]
{original_block}

{_output_instruction(output_format)}
""".strip()
