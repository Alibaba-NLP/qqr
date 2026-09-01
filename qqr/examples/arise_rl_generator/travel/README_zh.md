# ARISE-RL Generator — Travel（旅行规划出题者）

<h4 align="center">
    <p>
        <a href="README.md">English</a>&nbsp; | &nbsp;
        <b>中文</b>
    </p>
</h4>

多工具旅行规划任务的出题者训练。工具集：`poi_search / around_search /
direction / web_search / search_flights / search_train_tickets / weather`。
出题者输出 `query + expected_tools + rubrics`，其中 `expected_tools` 的
每个条目都必须是出题过程中真实调用过、且参数级校验通过的工具调用。

## 运行

```bash
bash scripts/arise_rl_generator/travel/start_executor_server.sh
bash scripts/arise_rl_generator/travel/run-qwen3.5-9B.sh
```

## 奖励设计（论文 Eq.(1)–(5)）

```
R_G = R_tool · R_fmt · (1 + R_diff)
R_tool = 1[出题轨迹包含真实工具观测，且 expected_tools 参数级校验通过]
R_fmt  = 1[query 非空 ∧ rubrics 为非空列表]
R_diff = 2 · max(0, 1 − |c − K/2| / (K/2)),  c = Σᵢ 1(rᵢ ≥ γ)
rᵢ     = α·sᵢ + (1−α)·1[sᵢ=1]   （做题者第 i 次试验的奖励）
```

默认超参（与论文一致）：K=8 次做题者验证，γ=0.9，α=0.8。
出题者被引导生成落在做题者能力边界附近（c ≈ K/2）的中等难度任务。

## RG-SED（代码中记为 RG-KL）

三阶段 rollout：student（无 coach memory）→ Coach LLM 生成组级 memory →
teacher（同一策略 + memory，仅用于估计 Δ_r）。
`λ_t = λ_0 · g(Δ_r) · w(t) · d(t)`，其中 `g(Δ_r) = σ((Δ_r − τ)/T)`，
默认 λ_0=0.5、τ=0.05、T=0.0125，cosine warm-up + cosine decay。
仅当 memory 带来经验奖励提升时，才通过 token 级 reverse KL 蒸馏回策略。
