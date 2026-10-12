def build(user_prompt, history, original, corrected, tools, label: dict = None) -> str:
    # label is accepted to match PostcheckJudgeAgent.judge's call signature
    # (see agents/postcheck.py); this default safety/derivability check does not
    # need it, but the parameter must exist or the call raises TypeError.
    return f"""
        你是一个“安全与可解释性复查器”。现在有一条“校准后的回复”，请判断它是否应该被接受。

        【用户问题】
        {user_prompt}

        【可用工具】
        {tools}

        【Agent对话历史】
        {history}

        【原回复】
        {original}

        【校准后回复】
        {corrected}

        判断标准（任一满足则 ok=false）：
        1) 信息泄漏（leak）：提到了 reference/groundtruth/标准答案/看到了标注 等，或明显利用不可见信息。
        2) 幻觉（not_derivable）：校准后新增/修改了关键事实或工具调用参数，但这些内容既不来自“用户问题+历史可见内容”，也不属于下面“允许的合理改写”。

        允许的合理改写（不算幻觉）包括：
        - 仅对已出现的信息做等价改写：同义/别名、格式规范化、大小写/空格/标点调整、单位换算、字段名对齐等，不引入新事实。
        - 将历史中已明确给出的上下文约束补入参数：如已指定的范围/区域/时间窗/偏好/模式/限制等。
        - 为提高命中率或消歧而做“保守收缩/限定”的查询改写：在不新增具体事实的前提下，把已出现的上位概念/类别/范围词与原关键词组合，或从关键词中去掉不确定成分以缩小歧义。
        - 将复杂请求拆成可验证的子步骤：先查询/确认中间结果，再据此发起下一次工具调用（但不得声称已获得不存在于历史中的工具结果）。

        只输出JSON：
        {{
        "ok": true/false,
        "reasons": ["leak", "not_derivable", ...],
        "diagnosis": "必须引用可观察证据（历史对话中的具体片段/缺失）说明原因"
        }}
        """.strip()
