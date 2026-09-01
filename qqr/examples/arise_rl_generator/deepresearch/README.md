# ARISE-RL Generator — DeepResearch

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

Generator training for the single-tool (`web_search`) deep-research task. The Generator first
explores a topic through real web_search calls, then authors `query + rubrics`, where every
rubric must be grounded in real search observations (tool-grounded rubric construction).

## Run

```bash
# 1. Start the Solver (executor) inference service
bash scripts/arise_rl_generator/deepresearch/start_executor_server.sh
# 2. Launch Generator training
bash scripts/arise_rl_generator/deepresearch/run-qwen3.5-9B.sh
```

## Reward Design (Eq. (1)-(5) in the paper)

```
R_G = R_tool * R_fmt * (1 + R_diff)
R_tool = 1[the authoring trajectory contains real tool observations]
R_fmt  = 1[query is non-empty AND rubrics is a non-empty list]
R_diff = 2 * max(0, 1 - |c - K/2| / (K/2)),  c = sum_i 1(r_i >= gamma)
r_i    = alpha*s_i + (1-alpha)*1[s_i=1]   (Solver reward of the i-th trial)
```

Default hyperparameters (following the paper): K=8 Solver verification trials, gamma=0.9, alpha=0.8.
The Generator is driven to author intermediate-difficulty tasks near the Solver's capability
boundary (c around K/2).

## RG-SED (referred to as RG-KL in the code)

Three-stage rollout: student (no coach memory) -> a Coach LLM summarizes group-level memory ->
teacher (same policy + memory, used only to estimate Delta_r).
`lambda_t = lambda_0 * g(Delta_r) * w(t) * d(t)` with `g(Delta_r) = sigmoid((Delta_r - tau)/T)`,
defaults lambda_0=0.5, tau=0.05, T=0.0125, cosine warm-up + cosine decay.
Distillation via token-level reverse KL is applied only when the memory empirically improves reward.
