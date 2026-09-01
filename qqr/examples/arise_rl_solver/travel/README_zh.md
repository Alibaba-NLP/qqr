# ARISE-RL Solver — Travel（旅行规划做题者）

<h4 align="center">
    <p>
        <a href="README.md">English</a>&nbsp; | &nbsp;
        <b>中文</b>
    </p>
</h4>

多工具旅行规划做题者训练。奖励同时考察内容 rubric 与过程级工具调用
（expected_tools 参数级匹配）：

```
s = (content_rubrics_met + tool_rubrics_met) / (content_total + n_expected_tools)
r = 0.8·s + 0.2·1[全部满足]      （论文 Eq.(6)，α=0.8）
```

RG-SED 设置与 deepresearch 相同（λ_0=0.5、τ=0.05、sigmoid gate）。

## 运行

```bash
bash scripts/arise_rl_solver/travel/run-qwen3.5-9B.sh
```

训练数据通过环境变量 `TRAIN_DATA` 传入（query+expected_tools+rubrics jsonl，
可由出题者 `arise_rl_generator/travel` 生成）。
评测：`data/ecr_travel/test.jsonl`（ECR-Travel，5 类子任务 × 100 条，
专家校准 rubric + expected_tools）。
