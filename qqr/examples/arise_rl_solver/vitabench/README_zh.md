# ARISE-RL Solver — VitaBench（交互式生活服务做题者）

<h4 align="center">
    <p>
        <a href="README.md">English</a>&nbsp; | &nbsp;
        <b>中文</b>
    </p>
</h4>

VitaBench 交互式做题者训练：Agent ↔ UserSimulator（Qwen3.5-397B）多轮
对话 + 工具调用，LLM Judge（gpt-5.2）按 rubric 评估。

## 奖励设计（论文 Eq.(6)）

```
训练:  r = 0.8·rubric_rate + 0.2·1[全通过]
评测:  pass^1 = 1[全通过]        （对齐 VitaBench 官方指标）
```

RG-SED 设置同其余任务（λ_0=0.5、τ=0.05、sigmoid gate、cosine warm-up/decay）。

## 数据与运行

环境与任务数据请从官方仓库
[meituan-longcat/vitabench](https://github.com/meituan-longcat/vitabench) 获取，
并设置 `VITABENCH_DATA_DIR` 指向其 `data/vita` 目录。

```bash
# VITABENCH_DOMAIN 可选 ota / delivery / instore / cross_domain
VITABENCH_DOMAIN=ota bash scripts/arise_rl_solver/vitabench/run-qwen3.5-9B.sh
```

做题者 system prompt 内置四个场景的专项规范（`DOMAIN_GUIDANCE`：ota / delivery /
instore / cross_domain），按任务 domain 自动注入；训练/评测数据（`TRAIN_DATA` /
`EVAL_DATA`）可由官方 `tasks.json` 转换，或由出题者
（`arise_rl_generator/vitabench`）生成。
