"""
ARISE-RL Solver（做题者）训练模块。

三个任务域的推理求解器（Solver），基于 rubric 满足信号 + RG-SED
（Reward-Gated Self-Evolution Distillation，代码中记为 RG-KL）训练：
- deepresearch: 单工具（web_search）深度研究
- travel:       多工具旅行规划
- vitabench:    VitaBench 交互式生活服务（UserSimulator 多轮对话）

做题者奖励（论文 Eq.(6)）：r = α·s + (1−α)·𝟙[s=1]，α=0.8，
s 为 rubric 满足率（travel/vitabench 同时计入工具调用 rubric）。
"""
