# ECR-Travel

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

The multi-tool travel-planning subset of ECR-Bench (Expert-Calibrated Rubric Benchmark).

| File | Size | Description |
|---|---|---|
| `test.jsonl` | 500 | ECR-Travel test set: 5 sub-tasks x 100 queries, perfectly balanced |

Five sub-tasks (`task_type`): `direction` (route planning with multiple waypoints),
`compare_itinerary` (transportation-mode comparison), `search_around` (nearby POI search),
`one_day_travel` (one-day itinerary), and `multi_day_travel` (multi-day itinerary).

## Format

```json
{"query": "...",
 "metadata": {
   "expected_tools": [{"name": "direction", "arguments": {"origin": "lon,lat", "...": "..."}}],
   "rubrics": ["...", "..."],
   "task_type": "direction"}}
```

Each query is paired with expert-calibrated rubrics (3-5 per query, median 4) and
`expected_tools` (rubric items specifying the required process-level tool-use behavior,
with arguments taken from real tool returns).
Aggregated expected_tools frequency over all 500 queries: direction 327, around_search 184,
poi_search 163, weather 135, search_flights 126, search_train_tickets 124.
Evaluation metric: task pass rate (all content rubrics and tool rubrics satisfied).
