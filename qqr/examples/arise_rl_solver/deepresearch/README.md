# ARISE-RL Solver — DeepResearch

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

Solver training for the single-tool (`web_search`) deep-research task: multi-round search +
structured research reports, scored rubric-by-rubric by an LLM judge (gpt-5.2).

## Reward Design (Eq. (6) in the paper)

```
r = alpha*s + (1-alpha)*1[s=1],  alpha = 0.8
```
`s` is the rubric satisfaction rate; for ResearchRubrics evaluation, compliance is computed
with the official rubric weights.

## RG-SED (referred to as RG-KL in the code)

Student rollouts join GRPO; teacher rollouts (same policy + coach memory) are used only to
estimate Delta_r. lambda_t = lambda_0 * sigmoid((Delta_r - tau)/T) * w(t) * d(t), defaults
lambda_0=0.5, tau=0.05. Teacher samples are reward-neutralized with loss_mask=0 so they do
not pollute GRPO normalization.

## Run

```bash
bash scripts/arise_rl_solver/deepresearch/run-qwen3.5-9B.sh
```

Training data is passed via the `TRAIN_DATA` environment variable (query+rubrics jsonl,
which can be produced by the Generator `arise_rl_generator/deepresearch`).
Evaluation: `data/ecr_deepresearch/test.jsonl` (ECR-DeepResearch, 100 expert-calibrated
rubric queries); to additionally evaluate ResearchRubrics, pass it via `RESEARCHRUBRICS_VAL`.
