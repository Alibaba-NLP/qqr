def build(user_prompt, history, original, corrected, tools, label) -> str:
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

        请按以下步骤判断校准后回复是否应被接受：

        步骤1：从【校准后回复】中提取所有坐标值（xxx.xxx,yyy.yyy格式）。若无坐标（如仅含poi_search重搜），跳过步骤2
        步骤2：对每个提取的坐标，在【历史工具返回】中查找同名地点的location字段。若坐标值在该地点的历史返回中不存在，则判not_derivable。注意：只检查你从【校准后回复】中实际提取到的坐标，不要检查【历史】或【原回复】中的坐标

        拒绝条件（任一满足则ok=false）：
        1) 信息泄漏（leak）：提到了reference/groundtruth/标准答案/参考信息/参考资料/正确答案/参考数据/根据参考/依据参考/参照参考/据参考/参考给出/参考提供
        2) 坐标造假（not_derivable）：步骤2发现坐标无历史来源

        允许的合法改写：同义/格式改写、拆分子步骤、改写为poi_search重搜

        只输出JSON：
        {{
        "ok": true/false,
        "reasons": ["leak", "not_derivable"],
        "coords_in_corrected": ["从校准后回复提取的坐标"] or "无",
        "diagnosis": "逐坐标溯源结果"
        }}
        """.strip()


# def build(user_prompt, history, original, corrected, tools) -> str:
#     return f"""
#         你是“校准回复复查器”。判断【校准后回复】是否仅由【可见信息】推出：用户问题+工具说明+对话历史(含工具返回)，除此之外一律不可用。

#         【用户问题】{user_prompt}
#         【工具】{tools}
#         【历史】{history}
#         【原回复】{original}
#         【校准后】{corrected}

#         拒绝(ok=false)任一满足：
#         1) leak：提到/暗示 reference、groundtruth、标准答案、标注、看过答案等。
#         2) not_derivable：校准后新增/改动的关键事实/数字/结论/工具参数在可见信息中无明确依据。
#         3) fabricated：声称工具返回了结果，但历史中无对应调用与返回。

#         允许：根据过往信息能够合理推出的更有效的推理回复。

#         只输出JSON：
#         {{"ok":true/false,"reasons":["leak","not_derivable","fabricated"],"diagnosis":"引用可见原文证据或指出缺失证据"}}
#         """.strip()
