# ECR-DeepResearch

<h4 align="center">
    <p>
        <a href="README.md">English</a>&nbsp; | &nbsp;
        <b>中文</b>
    </p>
</h4>

ECR-Bench 的单工具深度研究子集（Expert-Calibrated Rubric Benchmark）。

| 文件 | 条数 | 说明 |
|---|---|---|
| `test.jsonl` | 100 | ECR-DeepResearch 测试集：开放式研究 query + 专家校准 rubrics（每条 6–8 条，中位数 7） |

## 格式

```json
{"query": "...", "metadata": {"rubrics": ["可独立判定的断言 1", "断言 2", ...]}}
```

评测指标：rubric score ratio（LLM Judge 按 rubric 逐条判定的通过率）。
rubric 覆盖事实依据、证据覆盖、推理质量、完整性与报告结构五个维度，
全部经人工审核校准（详见论文 ECR-Bench 章节与附录）。
