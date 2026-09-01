# ARISE-RL Solver — DeepResearch（深度研究做题者）

<h4 align="center">
    <p>
        <a href="README.md">English</a>&nbsp; | &nbsp;
        <b>中文</b>
    </p>
</h4>

单工具（`web_search`）深度研究做题者训练：多轮搜索 + 结构化研究报告，
由 LLM Judge（gpt-5.2）按 rubric 逐条评估。

## 奖励设计（论文 Eq.(6)）

```
r = α·s + (1−α)·1[s=1],  α = 0.8
```
`s` 为 rubric 满足率；ResearchRubrics 评测时按官方权重计算 compliance。

## RG-SED（代码中记为 RG-KL）

student rollout 参与 GRPO；teacher（同一策略 + coach memory）仅用于
估计 Δ_r。λ_t = λ_0·σ((Δ_r−τ)/T)·w(t)·d(t)，默认 λ_0=0.5、τ=0.05。
Teacher 样本 reward 中性化 + loss_mask=0，不污染 GRPO 归一化。

## 运行

```bash
bash scripts/arise_rl_solver/deepresearch/run-qwen3.5-9B.sh
```

训练数据通过环境变量 `TRAIN_DATA` 传入（query+rubrics jsonl，可由出题者
`arise_rl_generator/deepresearch` 生成）。
评测：`data/ecr_deepresearch/test.jsonl`（ECR-DeepResearch，100 条专家校准 rubric）；
如需一并评测 ResearchRubrics，通过 `RESEARCHRUBRICS_VAL` 传入。
