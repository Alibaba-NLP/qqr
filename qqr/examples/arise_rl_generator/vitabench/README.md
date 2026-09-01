# ARISE-RL Generator — VitaBench

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

Generator training for interactive tasks on the official VitaBench OTA environments. The
Generator explores the closed simulation through real OTA tools and authors second-person
`instructions + rubrics`; the Solver verifies task difficulty in interactive mode
(Agent <-> UserSimulator <-> tools).

## Data & Run

Obtain the environment and task data from the official repository
[meituan-longcat/vitabench](https://github.com/meituan-longcat/vitabench) and set
`VITABENCH_DATA_DIR` to its `data/vita` directory.

```bash
# 1. Start the Solver (executor) inference service
bash scripts/arise_rl_generator/vitabench/start_executor_server.sh
# 2. Launch Generator training (VITABENCH_DOMAIN: ota / delivery / instore / cross_domain)
VITABENCH_DOMAIN=ota bash scripts/arise_rl_generator/vitabench/run-qwen3.5-9B.sh
```

## Scenario Coverage (alternative prompts)

The authoring system prompt, task types, and tool-chain mappings automatically switch
with `VITABENCH_DOMAIN`, covering all four VitaBench scenarios:

| Scenario | Task types | Core tool chains |
|---|---|---|
| `ota` | hotel / flight / train ticket / attraction ticket | `*_search_recommend -> get_ota_*_info -> create/pay` |
| `delivery` | food ordering / re-order from history / delivery ETA / order management | `delivery_*_search_recommend -> get_delivery_*_info -> create/pay_delivery_order`, ETA computation |
| `instore` | in-store deals / restaurant booking / service reservation | `instore_*_search_recommend -> create/pay_instore_*`, `instore_book`, `instore_reservation` |
| `cross_domain` | composite tasks spanning at least two domains | cross-domain combinations of the above |

Each seed line of `TRAIN_DATA` looks like
`{"query": "...", "metadata": {"task_id": "<official env id>", "domain": "<scenario>"}}`.

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
