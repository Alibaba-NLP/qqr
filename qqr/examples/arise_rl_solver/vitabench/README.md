# ARISE-RL Solver — VitaBench

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

Solver training on interactive VitaBench tasks: Agent <-> UserSimulator (Qwen3.5-397B)
multi-turn dialogue + tool calls, scored against rubrics by an LLM judge (gpt-5.2).

## Reward Design (Eq. (6) in the paper)

```
training:    r = 0.8*rubric_rate + 0.2*1[all passed]
evaluation:  pass^1 = 1[all passed]        (aligned with the official VitaBench metric)
```

RG-SED settings are the same as the other tasks (lambda_0=0.5, tau=0.05, sigmoid gate,
cosine warm-up/decay).

## Data & Run

Obtain the environment and task data from the official repository
[meituan-longcat/vitabench](https://github.com/meituan-longcat/vitabench) and set
`VITABENCH_DATA_DIR` to its `data/vita` directory.

```bash
# VITABENCH_DOMAIN: ota / delivery / instore / cross_domain
VITABENCH_DOMAIN=ota bash scripts/arise_rl_solver/vitabench/run-qwen3.5-9B.sh
```

The Solver system prompt ships domain-specific guidance for all four scenarios
(`DOMAIN_GUIDANCE`: ota / delivery / instore / cross_domain), injected automatically by
the task's domain. Training/evaluation data (`TRAIN_DATA` / `EVAL_DATA`) can be converted
from the official `tasks.json`, or produced by the Generator
(`arise_rl_generator/vitabench`).
