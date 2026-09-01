"""
Data preprocessing: convert the source JSONL into the format the QQR training framework expects

Source format (train_qqw_rubrics.jsonl):
  {"query": "...", "expected_tools": [...], "messages": [...], "rubrics": [...]}

Source format (test_qqw_rubrics.jsonl):
  {"query": "...", "task_type": "...", "metadata": {"expected_tools": [...]}, "reference": [...], "messages": [...], "rubrics": [...]}

Target format (unified):
  {"query": "...", "metadata": {"expected_tools": [...], "rubrics": [...]}}

Usage:
  python -m qqr.examples.travel_rubrics.prepare_data \
    --input /path/to/train_qqw_rubrics.jsonl \
    --output /path/to/train_qqw_rubrics_prepared.jsonl

  python -m qqr.examples.travel_rubrics.prepare_data \
    --input /path/to/test_qqw_rubrics.jsonl \
    --output /path/to/test_qqw_rubrics_prepared.jsonl
"""

import argparse
import json
import sys


def convert_record(data: dict) -> dict:
    """Convert a single JSONL record into the QQR framework format."""
    query = data.get("query", "")

    # extract expected_tools, preferring the top level and falling back to metadata
    expected_tools = data.get("expected_tools", [])
    if not expected_tools and isinstance(data.get("metadata"), dict):
        expected_tools = data["metadata"].get("expected_tools", [])

    # extract rubrics from the top level
    rubrics = data.get("rubrics", [])

    # extract task_type (optional)
    task_type = data.get("task_type", "")

    metadata = {
        "expected_tools": expected_tools,
        "rubrics": rubrics,
    }
    if task_type:
        metadata["task_type"] = task_type

    return {
        "query": query,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="转换 JSONL 数据为 QQR 训练格式")
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    args = parser.parse_args()

    count = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            converted = convert_record(data)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    print(f"转换完成: {count} 条记录, {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
