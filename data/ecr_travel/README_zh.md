# ECR-Travel

<h4 align="center">
    <p>
        <a href="README.md">English</a>&nbsp; | &nbsp;
        <b>中文</b>
    </p>
</h4>

ECR-Bench 的多工具旅行规划子集（Expert-Calibrated Rubric Benchmark）。

| 文件 | 条数 | 说明 |
|---|---|---|
| `test.jsonl` | 500 | ECR-Travel 测试集：5 类子任务 × 100 条，均衡分布 |

五类子任务（`task_type`）：`direction`（多途经点路线规划）、
`compare_itinerary`（交通方式比较）、`search_around`（周边 POI 搜索）、
`one_day_travel`（单日行程）、`multi_day_travel`（多日行程）。

## 格式

```json
{"query": "...",
 "metadata": {
   "expected_tools": [{"name": "direction", "arguments": {"origin": "lon,lat", "...": "..."}}],
   "rubrics": ["...", "..."],
   "task_type": "direction"}}
```

每条 query 配专家校准 rubrics（3–5 条，中位数 4）与 `expected_tools`
（规定过程级工具调用行为的 rubric 项，参数取自真实工具返回值）。
全量 500 条的 expected_tools 工具频次：direction 327、around_search 184、
poi_search 163、weather 135、search_flights 126、search_train_tickets 124。
评测指标：task pass rate（内容 rubric 与工具 rubric 全部满足）。
