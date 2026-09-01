# ARISE-RL Solver — Travel

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

Solver training for the multi-tool travel-planning task. The reward jointly assesses content
rubrics and process-level tool calls (argument-level matching of expected_tools):

```
s = (content_rubrics_met + tool_rubrics_met) / (content_total + n_expected_tools)
r = 0.8*s + 0.2*1[all satisfied]      (Eq. (6) in the paper, alpha=0.8)
```

RG-SED settings are the same as deepresearch (lambda_0=0.5, tau=0.05, sigmoid gate).

## Run

```bash
bash scripts/arise_rl_solver/travel/run-qwen3.5-9B.sh
```

Training data is passed via the `TRAIN_DATA` environment variable
(query+expected_tools+rubrics jsonl, which can be produced by the Generator
`arise_rl_generator/travel`).
Evaluation: `data/ecr_travel/test.jsonl` (ECR-Travel, 5 sub-tasks x 100 queries with
expert-calibrated rubrics + expected_tools).
