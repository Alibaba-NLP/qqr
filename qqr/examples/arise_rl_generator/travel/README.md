# ARISE-RL Generator — Travel

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

Generator training for the multi-tool travel-planning task. Tool set: `poi_search /
around_search / direction / web_search / search_flights / search_train_tickets / weather`.
The Generator emits `query + expected_tools + rubrics`, where every entry in `expected_tools`
must be a tool call actually made during authoring and validated at the argument level.

## Run

```bash
bash scripts/arise_rl_generator/travel/start_executor_server.sh
bash scripts/arise_rl_generator/travel/run-qwen3.5-9B.sh
```

## Reward Design (Eq. (1)-(5) in the paper)

```
R_G = R_tool * R_fmt * (1 + R_diff)
R_tool = 1[the authoring trajectory contains real tool observations, and expected_tools passes argument-level validation]
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
