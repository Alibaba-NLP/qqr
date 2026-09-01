"""
ARISE-RL Generator（出题者）训练模块。

三个任务域的任务/rubric 生成器（Generator），通过 rubric 介导的
Generator–Solver 协同演化训练（详见论文 ARISE-RL）：
- deepresearch: 单工具（web_search）深度研究出题
- travel:       多工具旅行规划出题（expected_tools 参数级校验）
- vitabench:    VitaBench 交互式生活服务出题

出题者奖励（论文 Eq.(1)-(5)）：R_G = R_tool · R_fmt · (1 + R_diff)，
其中 R_diff 为三角难度奖励，峰值位于做题者 K 次试验成功数 c = K/2 处。
"""
