# eval_test.py
import argparse
import asyncio
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

from qqr.examples.navigation import config
from qqr.examples.navigation.label_updater import update_label_locations
from qqr.examples.navigation.reward_model import (
    _extract_direction_calls_from_messages,
    _extract_groundtruth_from_label,
    _extract_last_tool_call_from_messages,
    _parse_order_from_label,
    navigation_reward,
)

from qqr.schemas import Sample
HAS_QQR = True
ITER_RE = re.compile(r"iter_(\d+)$")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _make_sample_from_row(row: Dict[str, Any]) -> Sample:
    label = row.get("label")
    messages = row.get("messages") or row.get("conversation") or []
    return Sample(label=label, messages=messages)


def _group_rows(rows: List[Dict[str, Any]], n_group: int) -> List[List[Dict[str, Any]]]:
    # 丢弃最后不足 n_group 的残组
    n = (len(rows) // n_group) * n_group
    rows = rows[:n]
    return [rows[i:i + n_group] for i in range(0, n, n_group)]


def _compute_pass_metrics(successes: List[int], ks: List[int]) -> Dict[str, float]:
    """
    successes: 每组的成功序列，如 [1,0,1,0]，长度>=max(ks)
    """
    out = {}
    for k in ks:
        prefix = successes[:k]
        out[f"pass@{k}"] = 1.0 if any(prefix) else 0.0
    for k in ks:
        prefix = successes[:k]
        out[f"pass^{k}"] = 1.0 if all(prefix) else 0.0
    return out


def _eval_one_file(test_jsonl: str, n_group: int = 4, use_merged_direction: bool = True) -> Tuple[Dict[str, Any], List[str]]:
    """
    评估单个文件
    
    Args:
        test_jsonl: 测试文件路径
        n_group: 每组样本数量
        use_merged_direction: 是否使用合并的 direction 调用逻辑
    """
    rows = _read_jsonl(test_jsonl)

    total = 0
    correct = 0
    reasons_head: List[str] = []

    # 分组统计
    groups = _group_rows(rows, n_group=n_group)
    n_groups = len(groups)
    ks = list(range(1, n_group + 1))
    pass_sums = {f"pass@{k}": 0.0 for k in ks}
    pass_sums.update({f"pass^{k}": 0.0 for k in ks})

    row_idx = 0
    for g in groups:
        group_successes: List[int] = []

        for _ in g:
            row = rows[row_idx]
            sample = _make_sample_from_row(row)

            try:
                asyncio.run(update_label_locations(sample))
            except Exception:
                pass

            # 使用新的合并逻辑或旧逻辑
            if use_merged_direction:
                direction_calls = _extract_direction_calls_from_messages(sample.messages)
                if direction_calls:
                    pred_nav = direction_calls[-1]  # 使用最后一个合并后的调用
                else:
                    pred_nav = None
            else:
                pred_nav = _extract_last_tool_call_from_messages(sample.messages, config.navigation_reward_name)
            
            gold_nav = _extract_groundtruth_from_label(sample.label)
            order = _parse_order_from_label(sample.label)

            total += 1
            if pred_nav is None or gold_nav is None:
                r = 0
                reason = "missing pred_nav or gold_nav"
            else:
                r, reason = navigation_reward(pred_nav=pred_nav, gold_nav=gold_nav, order=order)

            if r == 1:
                correct += 1

            if len(reasons_head) < 5 and r == 0:
                reasons_head.append(f"idx={row_idx}: {reason}")

            group_successes.append(1 if r == 1 else 0)
            row_idx += 1

        # 该组 pass@k / pass^k
        m = _compute_pass_metrics(group_successes, ks)
        for k, v in m.items():
            pass_sums[k] += v

    acc = (correct / total) if total > 0 else 0.0
    out = {
        "acc": acc,
        "correct": correct,
        "total": total,
        "n_group": n_group,
        "n_groups": n_groups,
    }

    if n_groups > 0:
        for k in ks:
            out[f"pass@{k}"] = pass_sums[f"pass@{k}"] / n_groups
            out[f"pass^{k}"] = pass_sums[f"pass^{k}"] / n_groups
    else:
        for k in ks:
            out[f"pass@{k}"] = 0.0
            out[f"pass^{k}"] = 0.0

    return out, reasons_head


def _eval_single_json(json_path: str, use_merged_direction: bool = True) -> Dict[str, Any]:
    """
    评估单个 JSON 文件（非 jsonl 格式）
    
    Args:
        json_path: JSON 文件路径
        use_merged_direction: 是否使用合并的 direction 调用逻辑
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建 Sample 对象
    label = data.get("label")
    messages = data.get("messages") or data.get("conversation") or []
    sample = Sample(label=label, messages=messages)
    
    try:
        asyncio.run(update_label_locations(sample))
    except Exception as e:
        print(f"Warning: update_label_locations failed: {e}")
    
    # 使用新的合并逻辑或旧逻辑
    if use_merged_direction:
        direction_calls = _extract_direction_calls_from_messages(sample.messages)
        if direction_calls:
            pred_nav = direction_calls[-1]  # 使用最后一个合并后的调用
            print(f"找到 {len(direction_calls)} 个合并后的 direction 调用")
            for i, call in enumerate(direction_calls):
                args = call["arguments"]
                print(f"  调用 {i+1}: {args.get('origin')} -> {args.get('destination')}")
                if args.get('waypoints'):
                    print(f"    途经点: {args['waypoints']}")
        else:
            pred_nav = None
            print("未找到 direction 调用")
    else:
        pred_nav = _extract_last_tool_call_from_messages(sample.messages, config.navigation_reward_name)
        if pred_nav:
            args = pred_nav["arguments"]
            print(f"找到 direction 调用: {args.get('origin')} -> {args.get('destination')}")
        else:
            print("未找到 direction 调用")
    
    gold_nav = _extract_groundtruth_from_label(sample.label)
    order = _parse_order_from_label(sample.label)
    
    print(f"\nGolden 标准:")
    if gold_nav:
        args = gold_nav["arguments"]
        print(f"  起点: {args.get('origin')}")
        print(f"  终点: {args.get('destination')}")
        print(f"  途经点: {args.get('waypoints', '无')}")
        print(f"  Order: {order}")
    else:
        print("  未找到 golden 标准")
    
    if pred_nav is None or gold_nav is None:
        r = 0
        reason = "missing pred_nav or gold_nav"
    else:
        r, reason = navigation_reward(pred_nav=pred_nav, gold_nav=gold_nav, order=order)
    
    print(f"\n评估结果:")
    print(f"  奖励: {r}")
    print(f"  原因: {reason}")
    print(f"  结果: {'✓ 正确' if r == 1 else '✗ 错误'}")
    
    return {
        "reward": r,
        "reason": reason,
        "correct": r == 1,
        "pred_nav": pred_nav,
        "gold_nav": gold_nav,
        "order": order,
    }


def _iter_dirs(eval_root: str) -> List[str]:
    dirs = []
    for p in glob.glob(os.path.join(eval_root, "iter_*")):
        if os.path.isdir(p) and ITER_RE.search(os.path.basename(p)):
            dirs.append(p)
    dirs.sort(key=lambda x: int(ITER_RE.search(os.path.basename(x)).group(1)))
    return dirs

def main():
    ap = argparse.ArgumentParser(description="评估导航任务结果")
    
    # 模式选择
    ap.add_argument("--mode", type=str, default="batch", 
                    choices=["batch", "single"],
                    help="评估模式: batch=批量评估目录, single=评估单个文件")
    
    # batch 模式参数
    ap.add_argument("--run_name", type=str, default="Qwen3.5-9B/Qwen3.5-9B_0-6",
                    help="运行名称（batch 模式）")
    # ap.add_argument("--base_dir", type=str, default="../../../outputs/navigation/ckpts",help="基础目录（batch 模式）")
    ap.add_argument("--base_dir", type=str, default="/mnt/nas-alinlp/linghe.cx/rollcab/qqw/.trash/navigation/ckpts",help="基础目录（batch 模式）")
    
    ap.add_argument("--split", type=str, default="navigation", choices=["test", "dev", "train", "navigation"],
                    help="数据分割（batch 模式）")
    
    # single 模式参数
    ap.add_argument("--input", type=str, default="navigation.jsonl",
                    help="输入文件路径（single 模式），支持 .json 或 .jsonl 文件")
    
    # 通用参数
    ap.add_argument("--n_group", type=int, default=4, 
                    help="每组样本数量（batch 模式）")
    ap.add_argument("--no_merge", action="store_true",
                    help="禁用 direction 调用合并逻辑（使用旧逻辑）")
    
    args = ap.parse_args()
    
    use_merged = not args.no_merge
    
    if args.mode == "single":
        # 单文件评估模式
        if not args.input:
            print("Error: --input 参数在 single 模式下是必需的")
            return
        
        if not os.path.exists(args.input):
            print(f"Error: 文件不存在: {args.input}")
            return
        
        print(f"评估文件: {args.input}")
        print(f"使用合并逻辑: {use_merged}")
        print("=" * 60)
        
        if args.input.endswith('.jsonl'):
            # 评估 jsonl 文件
            metrics, reasons_head = _eval_one_file(args.input, n_group=args.n_group, use_merged_direction=use_merged)
            print(f"\n评估结果:")
            print(f"  准确率: {metrics['acc']:.4f} ({metrics['correct']}/{metrics['total']})")
            print(f"  组数: {metrics['n_groups']}")
            for k in range(1, args.n_group + 1):
                print(f"  pass@{k}: {metrics.get(f'pass@{k}', 0):.4f}")
                print(f"  pass^{k}: {metrics.get(f'pass^{k}', 0):.4f}")
            
            if reasons_head:
                print(f"\n部分失败原因:")
                for reason in reasons_head:
                    print(f"  - {reason}")
        else:
            # 评估单个 json 文件
            result = _eval_single_json(args.input, use_merged_direction=use_merged)
            
            # 保存结果
            output_path = args.input.replace('.json', '_eval_result.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n详细结果已保存到: {output_path}")
    
    else:
        # 批量评估模式（原有逻辑）
        run_root = os.path.join(args.base_dir, args.run_name)
        eval_root = os.path.join(run_root, "rollout", "evaluation")
        if not os.path.isdir(eval_root):
            raise FileNotFoundError(f"evaluation dir not found: {eval_root}")

        iters = _iter_dirs(eval_root)
        if not iters:
            raise FileNotFoundError(f"no iter_* dirs under: {eval_root}")

        print(f"批量评估模式")
        print(f"使用合并逻辑: {use_merged}")
        print(f"找到 {len(iters)} 个迭代目录")
        print("=" * 60)

        summary = []
        for iter_dir in iters:
            test_path = os.path.join(iter_dir, f"{args.split}.jsonl")
            if not os.path.exists(test_path):
                continue

            metrics, reasons_head = _eval_one_file(test_path, n_group=args.n_group, use_merged_direction=use_merged)
            iter_id = int(ITER_RE.search(os.path.basename(iter_dir)).group(1))

            out = {
                "run_name": args.run_name,
                "iter": iter_id,
                "split": args.split,
                "tool_name": config.navigation_reward_name,
                "reward_mode": getattr(config, "reward_mode", None),
                **metrics,
                "sample_fail_reasons_head": reasons_head,
            }

            out_path = os.path.join(iter_dir, "acc.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            summary.append(out)
            
            print(f"Iter {iter_id}: acc={metrics['acc']:.4f}, correct={metrics['correct']}/{metrics['total']}")

        summary_path = os.path.join(eval_root, f"acc_{args.split}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n汇总结果已保存到: {summary_path}")


if __name__ == "__main__":
    main()
