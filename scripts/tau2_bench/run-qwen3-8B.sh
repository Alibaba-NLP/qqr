#!/bin/bash
# RTI (Rollout-Time Intervention) training launcher for the tau2-bench task
# (Qwen3-8B, telecom domain training + OOD eval, 8 GPUs / TIS).
# Entry point: python3 -m qqr.off_policy.train_async.
#
# User-simulator settings below are per-experiment, so they are set HERE (at
# the top of this script) rather than in ~/.bashrc:
#   USER_SIM_BASE_URL  - OpenAI-compatible endpoint for the user simulator LLM
#   USER_SIM_MODEL     - model name served at USER_SIM_BASE_URL (default below)
# Optional env vars:
#   USER_SIM_API_KEY   - default "EMPTY"
#   JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY
#                      - judge LLM for NL-assertion rewards; default to the
#                        user-simulator settings above
#   QQR_ROOT / SLIME_PATH / MEGATRON_LM_PATH - repo locations (auto-detected)
#   WANDB_API_KEY      - enables wandb logging

# ── user-simulator settings (edit here, not in ~/.bashrc) ────────────
USER_SIM_BASE_URL=${USER_SIM_BASE_URL:-"http://10.56.14.11:8000/v1"}
USER_SIM_MODEL=${USER_SIM_MODEL:-"Qwen3.5-397B-A17B"}

# ── RTI corrector/judge endpoint (edit here, not in ~/.bashrc) ───────
# Points the corrector/judge at a separately served endpoint instead of the
# SGLang policy being trained.
INSPECTOR_OPENAI_BASE_URL=${INSPECTOR_OPENAI_BASE_URL:-"http://10.56.14.11:8000/v1"}
INSPECTOR_OPENAI_MODEL=${INSPECTOR_OPENAI_MODEL:-"${USER_SIM_MODEL}"}
INSPECTOR_OPENAI_API_KEY=${INSPECTOR_OPENAI_API_KEY:-EMPTY}

missing=""
[ -z "${USER_SIM_BASE_URL}" ] && missing="${missing} USER_SIM_BASE_URL"
if [ -n "${missing}" ]; then
  echo "[ERROR] Missing required environment variables:${missing}" >&2
  echo "        Set USER_SIM_BASE_URL at the top of this script (per-experiment endpoint)." >&2
  exit 1
fi

USER_SIM_API_KEY=${USER_SIM_API_KEY:-EMPTY}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-${USER_SIM_BASE_URL}}
JUDGE_MODEL=${JUDGE_MODEL:-${USER_SIM_MODEL}}
JUDGE_API_KEY=${JUDGE_API_KEY:-${USER_SIM_API_KEY}}

# ── cleanup stale processes from previous runs ──────────────────────
# Targeted kills only: never run a blanket `pkill python`, which would take
# down unrelated services running on this machine.
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 -f '/ray/' 2>/dev/null || true
ps -ef | grep -E 'python(3)? .*train(_async)?\.py|qqr\.off_policy\.train_async' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 3
pkill -9 -f '/ray/' 2>/dev/null || true
rm -rf /tmp/ray/tau2-tel-8b-async-ood-tis

set -eo pipefail

# ── resolve repo locations ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QQR_PATH=${QQR_ROOT:-$(pip list 2>/dev/null | grep -E '^qqr[[:space:]]' | awk '{print $NF}' || true)}
if [ -z "${QQR_PATH}" ]; then
  # qqr is not necessarily pip-installed; fall back to the repo rooted two
  # levels above this script (scripts/<task>/run-*.sh).
  QQR_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
if [ ! -f "${QQR_PATH}/qqr/args.py" ]; then
  echo "[ERROR] QQR_PATH=${QQR_PATH} does not look like the qqr repo" >&2
  exit 1
fi

SLIME_PATH=${SLIME_PATH:-$(pip list 2>/dev/null | grep -E '^slime[[:space:]]' | awk '{print $NF}' || true)}
if [ -z "${SLIME_PATH}" ] || [ ! -d "${SLIME_PATH}/slime" ]; then
  echo "[ERROR] Cannot resolve SLIME_PATH; activate the venv with the editable" >&2
  echo "        slime install (or export SLIME_PATH=/path/to/slime)." >&2
  exit 1
fi
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/root/Megatron-LM}

cd ${SLIME_PATH}

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23567}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}

export PYTHONUNBUFFERED=1
export PYTHONPATH=${QQR_PATH}:${SLIME_PATH}:${MEGATRON_LM_PATH}:${PYTHONPATH:-}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=${HAS_NVLINK}
export NCCL_DEBUG=WARN

export RAY_NUM_SERVER_CALL_THREAD=1
export TOKENIZERS_PARALLELISM=false
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

source "${SLIME_PATH}/scripts/models/qwen3-8B.sh"

CKPT_DIR="${QQR_PATH}/outputs/tau2_bench-telecom/ckpts/qwen3-8b/qwen3-8b-async-ood-tis"

CKPT_ARGS=(
   --hf-checkpoint /mnt/nas-alinlp/linghe.cx/models/Qwen/Qwen3-8B
   --ref-load /mnt/nas-alinlp/linghe.cx/models_dist/Qwen/Qwen3-8B_torch_dist
   --load ${CKPT_DIR}
   --save ${CKPT_DIR}
   --save-interval 5
   --save-rollout-data
   --normalize-advantages
)

ROLLOUT_ARGS=(
   --rollout-function-path qqr.rollout.agent_rollout.generate_rollout
   --prompt-data ${QQR_PATH}/data/tau2_bench/train_telecom.jsonl
   --input-key query
   --label-key reference
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-context-len 40000
   --rollout-max-response-len 9096
   --rollout-temperature 0.8
   --rollout-max-turns 40

   --global-batch-size 128
   --balance-data
   --group-rm

   --rollout-all-samples-process-path qqr.examples.tau2_bench.save_rollout_data.save_train_samples
   # RTI two-stage group rollout: Stage-1 normal rollout (inspector off); if the
   # whole group fails, Stage-2 re-runs with the inspector on and replaces
   # `inspector-replace-num` failed samples with successful calibrated ones.
   # Evaluation always stays in Stage-1.
   --custom-group-rollout-function-path qqr.off_policy.group_rollout.generate_and_rm_group_with_calibration
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
   --tis-clip-low 0.8
   --tis-clip 1.2
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 4

   --use-dynamic-batch-size
   --max-tokens-per-gpu 24576
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data tau2_bench ${QQR_PATH}/data/tau2_bench/test_telecom.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-context-len 40000
   --eval-input-key query
   --eval-label-key reference
   --custom-eval-rollout-log-function-path qqr.examples.tau2_bench.eval_rollout.custom_eval_rollout_log_function
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.5
   --sglang-server-concurrency 8
   --sglang-reasoning-parser qwen3
   --sglang-tool-call-parser qwen3_coder
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

RTI_ARGS=(
   --inspector-enabled
   --inspector-ratio 1.0
   --inspector-log-dir ${QQR_PATH}/outputs/tau2_bench-telecom/inspector/qwen3-8b/qwen3-8b-async-ood-tis
   --inspector-use-judge-llm
   --inspector-temperature 0.3
   --inspector-max-new-tokens 9096
   --enable-postcheck
   --inspector-fallback-enabled
   --inspector-replace-num 1
   --custom-inspector-prompt-path qqr.examples.tau2_bench.inspector_prompts

   --calc-ref-prob-diff
   --enable-ref-kl-gate
   --ref-judge-method ppl
   --ref-ppl-threshold 20.0
   --ref-js-threshold 0.3
   --ref-kl-threshold 1.0
   --ref-kl-temperature 1.0
   --ref-kl-min-prefix-tokens 0

   --save-rollout-data
   --save-inspector-data
)
# Corrector/judge endpoint (defaults set at the top of this script).
RTI_ARGS+=(
   --inspector-openai-base-url "${INSPECTOR_OPENAI_BASE_URL}"
   --inspector-openai-model "${INSPECTOR_OPENAI_MODEL}"
   --inspector-openai-api-key "${INSPECTOR_OPENAI_API_KEY}"
)

CUSTOM_ARGS=(
   --custom-tis-function-path qqr.reward_models.utils.reshape_gt_tis_function
   --custom-generate-function-path qqr.examples.tau2_bench.generate
   --custom-rm-path qqr.examples.tau2_bench.group_reward
   --custom-reward-post-process-path qqr.examples.tau2_bench.reward_post_process

   --user-sim-openai-base-url ${USER_SIM_BASE_URL}
   --user-sim-openai-api-key ${USER_SIM_API_KEY}
   --user-sim-model ${USER_SIM_MODEL}
   --user-sim-temperature 0
   --user-sim-max-new-tokens 4096
   --user-sim-timeout 180

   --judge-openai-base-url ${JUDGE_BASE_URL}
   --judge-openai-api-key ${JUDGE_API_KEY}
   --judge-model ${JUDGE_MODEL}
   --judge-timeout 60
)

WANDB_ARGS=(
   --wandb-project tau2-bench-grpo-rti
   --wandb-group qwen3-8b-async-ood-tis
   --disable-wandb-random-suffix
   --log-multi-turn
)
# Enable wandb only when a key is provided (the key itself is never hardcoded).
if [ -n "${WANDB_API_KEY:-}" ]; then
   WANDB_ARGS+=(--use-wandb --wandb-key "${WANDB_API_KEY}")
fi

if [ $RANK -eq 0 ]; then

ray start --head --port 6493 --num-gpus 8 --metrics-export-port 9102 \
   --node-ip-address 127.0.0.1 --disable-usage-stats \
   --temp-dir /tmp/ray/tau2-tel-8b-async-ood-tis

python3 -m qqr.off_policy.train_async \
   --actor-num-nodes ${NNODES} \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RTI_ARGS[@]} \
   ${CUSTOM_ARGS[@]}

sleep 30m

else

echo "Starting Ray worker to ${MASTER_ADDR}"
ray start --block --address ${MASTER_ADDR}:6493 --num-gpus 8 --metrics-export-port 9102 \
   --disable-usage-stats --temp-dir /tmp/ray/tau2-tel-8b-async-ood-tis

fi
