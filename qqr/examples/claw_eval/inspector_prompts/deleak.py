"""deleak：把可能泄漏的推理草稿改写成智能体自然可产出的推理。

与共享 RTI 流水线接口一致（agents/deleak.py）：

    build(history, draft_reasoning, draft_calls)

这是校准链路里的“去泄漏”环节：在保持工具调用完全不变的前提下，把 corrector 草稿里的推理重写成
“一个没看过任何反馈 / 参考 / 答案的智能体，仅凭可见历史就能自然写出的推理”。专门清除
“根据反馈 / Based on feedback / 参考答案 / 已知……应当” 这类痕迹。
"""


def build(history, draft_reasoning, draft_calls) -> str:
    return f"""
请把下面这段“可能带泄漏痕迹”的推理草稿，改写成智能体在**不知道任何参考 / groundtruth / 反馈 / 评估标准**
的情况下，仅凭可见历史就能自然产出的推理。

必须满足：
1) 只使用[可观察历史]中明确出现的信息（含其中的工具返回）。不得引入历史中无法推出的新事实、数值或结论。
2) 去掉任何“我已经知道答案 / 根据反馈 / 根据参考 / 评估要求 / 我应当调用 X 因为它是必需的”这类口吻；
   不要出现“根据反馈”“Based on feedback”“参考答案”“标准答案”“评估标准”“按照要求”等措辞；
   不要用“问题识别：1)… 2)…”“**FIX：**”这种照抄监督清单式的开头。
   没有可见依据就不要下确定性结论，改用前瞻性的表达（如“我先核实一下…”“我需要确认…”）。
3) 改写后的推理必须自然导向给定的目标工具调用：说明最小意图（要查什么/做什么、为什么），
   语言、风格与原草稿一致。**不得改变、增删或重排任何工具调用。**
4) 若草稿引用了历史中不可见的某个“必需动作/标准”，请改写成基于用户请求与既有工具返回的自然推断。

[可观察历史]
{history}

[可能泄漏的推理草稿]
{draft_reasoning}

[目标工具调用（保持不变，禁止修改）]
{draft_calls}

只输出改写后的推理正文（不要复述工具调用，不要加解释性前后缀）。
""".strip()
"""deleak: rewrite a reasoning draft to remove any groundtruth/reference leakage.

Adapted from navigation/inspector_prompts/deleak.py for the claw-eval domain.
"""


def build(history, draft_reasoning, draft_calls) -> str:
    return f"""
        你要把一段可能包含“标准答案/参考信息泄漏”的推理，改写为“claw-eval 智能体在不知道标准答案的情况下也能自然说出的推理”。

        必须满足：
        - 所有信息必须来自历史消息或历史工具返回（服务查询结果、记录、邮件、日程等）。
        - 不得出现“参考/groundtruth/技能显示/我知道正确做法是/标准答案”等任何暗示先验知识的表述。
        - 推理要能自然解释【目标工具调用】的动机与参数来源。
        - 只输出改写后的 reasoning 纯文本（不要 JSON、不要工具调用），不得修改目标工具调用本身。

        【历史对话（可观察信息）】
        {history}

        【可能泄漏的推理草稿】
        {draft_reasoning}

        【目标工具调用（保持不变）】
        {draft_calls}
    """.strip()
