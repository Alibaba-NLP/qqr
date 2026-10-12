# qqr/examples/navigation/eval_rollout.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qqr.schemas import Sample
from . import reward_model  # 直接复用你写好的reward解析逻辑

logger = logging.getLogger(__name__)


def _compute_pointwise_correct_for_sample(sample: Sample) -> int:
    """
    不看 sample.reward，直接用 reward_model 中的解析逻辑判断该样本是否“正确”。

    这里的“正确”定义为：
      navigation_reward(pred_nav, gold_nav, order) 返回的 r == 1
    """
    # 直接使用 reward_model 中的提取函数
    pred_nav = reward_model._extract_last_tool_call_from_messages(
        sample.messages, reward_model.config.navigation_reward_name
    )
    if pred_nav is None:
        # 没有预测到对应工具调用 -> 记为错误
        return 0

    gold_nav = reward_model._extract_groundtruth_from_label(sample.label)
    if gold_nav is None:
        # 没有 gold -> 记为错误
        return 0

    order = reward_model._parse_order_from_label(sample.label)
    r, _ = reward_model.navigation_reward(
        pred_nav=pred_nav, gold_nav=gold_nav, order=order
    )
    return int(r)  # r 是 0/1


def _extract_flat_correct_and_group_size(
    args, data: Dict[str, Any]
) -> Tuple[List[int], int]:
    """
    从 eval data 中抽取“是否正确”的0/1序列和 group_size（= n_samples_per_eval_prompt）。

    data 结构来自 RolloutManager._save_rollout_data 中 evaluation=True 的分支：
        data[dataset_name] = {
            "samples": List[Sample],
            "rewards": List[float],
            ...
        }

    我们忽略 rewards，重新对 samples 做判断。
    """
    n_samples_per_eval_prompt = args.n_samples_per_eval_prompt
    flat_correct: List[int] = []

    for dataset_name, info in data.items():
        samples: List[Sample] = info["samples"]
        logger.info(
            "Computing correctness for eval dataset '%s', num_samples=%d",
            dataset_name,
            len(samples),
        )
        for s in samples:
            c = _compute_pointwise_correct_for_sample(s)
            flat_correct.append(c)

    if len(flat_correct) % n_samples_per_eval_prompt != 0:
        logger.warning(
            "Total eval samples (%d) is not divisible by n_samples_per_eval_prompt (%d). "
            "Pass@k grouping may be inaccurate.",
            len(flat_correct),
            n_samples_per_eval_prompt,
        )

    return flat_correct, n_samples_per_eval_prompt


def _compute_pass_metrics_from_correct(flat_correct: List[int], group_size: int) -> Dict[str, float]:
    """
    基于"是否正确"的0/1序列计算 pass@k 和 pass^k，k=1..group_size。

    假设 flat_correct 已按 prompt-group 顺序排列：
        [p0_s0, p0_s1, ..., p0_s[group_size-1],
         p1_s0, p1_s1, ..., p1_s[group_size-1],
         ...]

    pass@k：对每个 prompt，取前 k 个样本，至少有一个成功则计 1（any）。
    pass^k：对每个 prompt，取前 k 个样本，全部成功才计 1（all）。
    最终取所有 prompt 的均值。
    """
    num_prompts = len(flat_correct) // group_size
    if num_prompts == 0:
        result: Dict[str, float] = {}
        for k in range(1, group_size + 1):
            result[f"pass@{k}"] = 0.0
            result[f"pass^{k}"] = 0.0
        return result

    pass_at_k_sums: Dict[int, float] = {k: 0.0 for k in range(1, group_size + 1)}
    pass_hat_k_sums: Dict[int, float] = {k: 0.0 for k in range(1, group_size + 1)}

    for i in range(num_prompts):
        group = flat_correct[i * group_size : (i + 1) * group_size]
        for k in range(1, group_size + 1):
            prefix = group[:k]
            pass_at_k_sums[k] += 1.0 if any(c == 1 for c in prefix) else 0.0
            pass_hat_k_sums[k] += 1.0 if all(c == 1 for c in prefix) else 0.0

    metrics: Dict[str, float] = {}
    for k in range(1, group_size + 1):
        metrics[f"pass@{k}"] = pass_at_k_sums[k] / num_prompts
        metrics[f"pass^{k}"] = pass_hat_k_sums[k] / num_prompts

    return metrics



def _save_metrics_to_json(args, rollout_id: int, metrics: Dict[str, Any]):
    if not getattr(args, "save", None):
        logger.warning("args.save is not set; metric.json will not be written.")
        return

    base_dir = Path(args.save)
    out_dir = base_dir / "rollout" / "evaluation" / f"iter_{rollout_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metric.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Saved custom eval metrics to %s", out_path)


from qqr.utils.save_eval import save_eval_samples_jsonl as _save_eval_samples_jsonl


def custom_eval_rollout_log_function(
    rollout_id: int,
    args,
    data: Dict[str, Any],
    extra_metrics: Dict[str, Any] | None,
) -> bool:
    """
    自定义 eval 日志函数。

    参数签名需要与 _log_eval_rollout_data 中 load_function 调用保持一致：
        custom_log_func(rollout_id, args, data, extra_metrics)

    返回值:
        - True: 表示已完成日志记录，slime 不再执行默认 _log_eval_rollout_data 逻辑。
        - False: 表示使用默认逻辑记录日志。
    """
    if extra_metrics is None:
        extra_metrics = {}

    # 1. 从 data["samples"] 重新基于导航工具调用判断正确性，得到 flat_correct(0/1) + group_size
    flat_correct, group_size = _extract_flat_correct_and_group_size(args, data)

    # 2. 基于 correct(0/1) 计算 pass@k 和 pass^k
    pass_metrics = _compute_pass_metrics_from_correct(flat_correct, group_size)

    # 3. 可选：把 pass@k / pass^k 放入 extra_metrics（例如用于 wandb / tensorboard）
    for k, v in pass_metrics.items():
        extra_metrics[f"eval/{k}"] = v

    # 4. 生成一个包含 pass@k / pass^k 的精简字典写到 metric.json
    metric_to_save = {
        "rollout_id": rollout_id,
        "n_samples_per_eval_prompt": group_size,
        "num_samples": len(flat_correct),
        "num_prompts": len(flat_correct) // group_size,
    }
    metric_to_save.update(pass_metrics)

    _save_metrics_to_json(args, rollout_id, metric_to_save)
    _save_eval_samples_jsonl(args, rollout_id, data)

    # Inspector traces from eval samples (gated by save_inspector_data).
    from .save_rollout_data import _save_inspector_traces_from_samples
    for dataset_name, info in data.items():
        _save_inspector_traces_from_samples(
            args, rollout_id, info.get("samples", []),
            evaluation=True, dataset_name=dataset_name,
        )

    logger.info("Custom eval metrics (pass@k and pass^k based on tool correctness): %s", pass_metrics)

    # 返回 True 告诉上层：自定义 logging 已完成，不需要默认逻辑
    return True
