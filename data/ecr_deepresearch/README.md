# ECR-DeepResearch

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

The single-tool deep-research subset of ECR-Bench (Expert-Calibrated Rubric Benchmark).

| File | Size | Description |
|---|---|---|
| `test.jsonl` | 100 | ECR-DeepResearch test set: open-ended research queries + expert-calibrated rubrics (6-8 per query, median 7) |

## Format

```json
{"query": "...", "metadata": {"rubrics": ["independently verifiable assertion 1", "assertion 2", ...]}}
```

Evaluation metric: rubric score ratio (the fraction of rubrics judged as satisfied by an LLM judge).
Rubrics cover factual grounding, evidence coverage, reasoning quality, completeness, and report
structure, and are all manually reviewed and calibrated (see the ECR-Bench section and appendix
of the paper).
