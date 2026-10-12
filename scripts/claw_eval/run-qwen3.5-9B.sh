#!/bin/bash
# RTI (Rollout-Time Intervention) training launcher for claw-eval-hard
# (Qwen3.5-9B, 8 GPUs / TIS). Entry point: python3 -m qqr.off_policy.train_async.
#
# Required env vars:
#   DASHSCOPE_API_KEY   - official claw-eval grader judge (LLM_JUDGE_API_KEY is
#                         accepted as a fallback name below)
#   IDEALAB_API_KEY     - IdeaLab Google Search backend for the web-search tools
#   E2B_API_KEY         - E2B sandbox key (tasks run in E2B sandboxes)
#   CLAW_EVAL_ROOT      - claw-eval checkout (containing src/); optional when the
#                         claw_eval package is pip-installed in the venv
#
# Note: claw-eval-hard tasks do NOT use the simulated user agent (no task.yaml
# enables it), so USER_AGENT_BASE_URL is not required. If a future task set
# needs it, set USER_AGENT_BASE_URL right below (per-experiment endpoint).
# Optional env vars:
#   DASHSCOPE_BASE_URL  - default https://dashscope.aliyuncs.com/compatible-mode/v1
#   USER_AGENT_API_KEY / USER_AGENT_MODEL - default "EMPTY" / "Qwen3.5-397B-A17B"
#   GRADER_JUDGE_BASE_URL / GRADER_JUDGE_API_KEY / GRADER_JUDGE_MODEL
#                       - override the grader judge endpoint; leave unset to use
#                         the claw-eval checkout's config_general.yaml (qwen3.7-max)
#   E2B_DOMAIN          - required, exported in ~/.bashrc (no default here)
#   WANDB_API_KEY       - wandb key (never hardcoded; logging runs without it too)
#   INSPECTOR_OPENAI_BASE_URL / INSPECTOR_OPENAI_MODEL / INSPECTOR_OPENAI_API_KEY
#                       - external endpoint for the RTI (Stage-2) corrector/judge;
#                         defaults for 9B policies are set in this script

# Accept LLM_JUDGE_API_KEY as a fallback name for DASHSCOPE_API_KEY.
if [ -z "${DASHSCOPE_API_KEY:-}" ] && [ -n "${LLM_JUDGE_API_KEY:-}" ]; then
  export DASHSCOPE_API_KEY="${LLM_JUDGE_API_KEY}"
fi

# ── required env checks ──────────────────────────────────────────────
missing=""
[ -z "${DASHSCOPE_API_KEY:-}" ] && missing="${missing} DASHSCOPE_API_KEY"
[ -z "${IDEALAB_API_KEY:-}" ] && missing="${missing} IDEALAB_API_KEY"
[ -z "${E2B_API_KEY:-}" ] && missing="${missing} E2B_API_KEY"
if [ -n "${missing}" ]; then
  echo "[ERROR] Missing required environment variables:${missing}" >&2
  echo "        Export them (e.g. in ~/.bashrc) and re-run." >&2
  exit 1
fi

# ── RTI (Stage-2) corrector/judge endpoint (edit here, not in ~/.bashrc) ──
# Points the corrector/judge at a separately served endpoint instead of the
# SGLang policy being trained.
INSPECTOR_OPENAI_BASE_URL=${INSPECTOR_OPENAI_BASE_URL:-"http://10.56.2.37:8000/v1"}
INSPECTOR_OPENAI_MODEL=${INSPECTOR_OPENAI_MODEL:-"Qwen3.5-9B"}
INSPECTOR_OPENAI_API_KEY=${INSPECTOR_OPENAI_API_KEY:-EMPTY}

# CLAW_EVAL_ROOT is optional when the claw_eval package is installed in the venv.
if [ -n "${CLAW_EVAL_ROOT:-}" ]; then
  if [ ! -d "${CLAW_EVAL_ROOT}/src/claw_eval" ]; then
    echo "[ERROR] CLAW_EVAL_ROOT=${CLAW_EVAL_ROOT} does not contain src/claw_eval" >&2
    exit 1
  fi
elif ! python3 -c "import claw_eval" 2>/dev/null; then
  echo "[ERROR] CLAW_EVAL_ROOT is not set and \`python3 -c 'import claw_eval'\` failed" >&2
  echo "        (is the venv activated?). Export a checkout containing src/, e.g.:" >&2
  echo "        export CLAW_EVAL_ROOT=/path/to/claw-eval" >&2
  exit 1
fi

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
# Keep the ray temp dir name short: ray puts AF_UNIX sockets below it and the
# kernel caps socket paths at 107 bytes.
rm -rf /tmp/ray/claw95-rti-8gpu

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

# ── claw-eval environment ────────────────────────────────────────────
export CLAW_EVAL_FIXTURES_DIR=${CLAW_EVAL_FIXTURES_DIR:-${QQR_PATH}/data/claw-eval-hard/fixtures}
# claw-eval-hard task roots: 20 held-out eval tasks + 160 synthesized train tasks
export CLAW_EVAL_TASKS_DIR=${CLAW_EVAL_TASKS_DIR:-${QQR_PATH}/data/claw-eval-hard/tasks:${QQR_PATH}/data/claw-eval-hard/syn_tasks}
export DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
export SEARCH_CACHE_DIR=${SEARCH_CACHE_DIR:-/tmp/claw_eval_web_search_cache}
# Official claw-eval grader judge overrides (unset -> the checkout's
# config_general.yaml judge: DashScope qwen3.7-max with DASHSCOPE_API_KEY).
export GRADER_JUDGE_BASE_URL=${GRADER_JUDGE_BASE_URL:-}
export GRADER_JUDGE_API_KEY=${GRADER_JUDGE_API_KEY:-}
export GRADER_JUDGE_MODEL=${GRADER_JUDGE_MODEL:-}
export GRADER_JUDGE_CONCURRENCY=16
# Simulated user agent (unused by claw-eval-hard: no task enables it). The
# endpoint changes per experiment, so set these here rather than in ~/.bashrc:
# export USER_AGENT_BASE_URL="http://<user-agent-host>:8000/v1"
# export USER_AGENT_API_KEY=EMPTY
# export USER_AGENT_MODEL=Qwen3.5-397B-A17B
# E2B sandbox domain comes from the environment (~/.bashrc), never hardcoded.
export E2B_DOMAIN=${E2B_DOMAIN}
# Rollout sample dump target (offline inspection)
export TASK_NAME=Qwen3.5-9B-claw-eval-hard-grpo-rti-8gpus
# This experiment is unimodal: do not feed tool screenshots into the model.
export CLAW_EVAL_MULTIMODAL_INPUT_ENABLED=false
# Cap per-tool-call text inserted into the conversation so that massive
# OCR / document-extraction results cannot exhaust the rollout context.
export CLAW_TOOL_RESULT_MAX_CHARS=50000
# RTI Stage-2 self-correction (read by qqr.examples.claw_eval.config):
# replace up to RTI_REPLACE_NUM failed samples per group; a trajectory counts
# as a success only when its official-grader score >= RTI_SUCCESS_THRESHOLD.
export RTI_ENABLED=true
export RTI_REPLACE_NUM=2
export RTI_SUCCESS_THRESHOLD=0.75

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23575}
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

source "${QQR_PATH}/scripts/models/qwen3.5-9B.sh"

CKPT_DIR="${QQR_PATH}/outputs/claw-eval-hard/ckpts/qwen3.5-9b/qwen3.5-9b-grpo-rti-8gpus-tis"

CKPT_ARGS=(
   --hf-checkpoint /mnt/nas-alinlp/linghe.cx/models/Qwen/Qwen3___5-9B
   --ref-load /mnt/nas-alinlp/linghe.cx/models_dist/Qwen/Qwen3.5-9B_torch_dist
   --load ${CKPT_DIR}
   --save ${CKPT_DIR}
   --save-interval 5
   --save-rollout-data
)

ROLLOUT_ARGS=(
   --rollout-function-path qqr.rollout.agent_rollout.generate_rollout
   --prompt-data ${QQR_PATH}/data/claw-eval-hard/train.jsonl
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-context-len 80000
   --rollout-max-response-len 20000
   --rollout-temperature 0.8

   --global-batch-size 64
   --balance-data
   --group-rm

   --rollout-all-samples-process-path qqr.examples.claw_eval.save_rollout_data.save_train_samples
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
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.001
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
   --eval-prompt-data claw-eval-hard ${QQR_PATH}/data/claw-eval-hard/test.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-context-len 80000
   --eval-input-key prompt
   --eval-label-key label
   --custom-eval-rollout-log-function-path qqr.examples.claw_eval.eval_rollout.custom_eval_rollout_log_function
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.4
   --sglang-server-concurrency 16
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

CUSTOM_ARGS=(
   --custom-generate-function-path qqr.examples.claw_eval.rollout.generate
   # claw uses its own RTI two-stage group rollout (Stage-2 self-correction with
   # the shared inspector pipeline, gated by RTI_ENABLED / RTI_REPLACE_NUM /
   # RTI_SUCCESS_THRESHOLD above).
   --custom-group-rollout-function-path qqr.examples.claw_eval.group_rollout.generate_and_rm_group_with_rti
   --custom-rm-path qqr.examples.claw_eval.reward_model.group_reward_by_official_grader
   --custom-reward-post-process-path qqr.examples.claw_eval.reward_model.correct_group_norm
   --custom-tis-function-path qqr.reward_models.utils.reshape_gt_tis_function
)

# Stage-2 RTI inspector args. The generic inspector path itself stays disabled
# (QR group rollout above does the RTI); these feed the calibration pipeline.
INSPECTOR_ARGS=(
   --inspector-ratio 0.0
   --inspector-replace-num 0
   --inspector-log-dir ${QQR_PATH}/outputs/claw-eval-hard/inspector/qwen3.5-9b/qwen3.5-9b-grpo-rti-8gpus-tis
   --save-inspector-data
   --custom-inspector-prompt-path qqr.examples.claw_eval.inspector_prompts
   --inspector-use-judge-llm
   --inspector-temperature 0.3
   --inspector-max-new-tokens 9096
   --enable-postcheck

   --calc-ref-prob-diff
   --enable-ref-kl-gate
   --ref-judge-method ppl
   --ref-ppl-threshold 20.0
   --ref-js-threshold 0.3
   --ref-kl-threshold 1.0
   --ref-kl-temperature 1.0
   --ref-kl-min-prefix-tokens 0
)
# Corrector/judge endpoint (defaults set at the top of this script).
INSPECTOR_ARGS+=(
   --inspector-openai-base-url "${INSPECTOR_OPENAI_BASE_URL}"
   --inspector-openai-model "${INSPECTOR_OPENAI_MODEL}"
   --inspector-openai-api-key "${INSPECTOR_OPENAI_API_KEY}"
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project claw-eval-hard-grpo-rti
   --wandb-group qwen3.5-9b-grpo-rti-ppl_20-8gpus-tis
   --disable-wandb-random-suffix
   --log-multi-turn
)
# The wandb key itself is never hardcoded; pass it only when the env provides one.
if [ -n "${WANDB_API_KEY:-}" ]; then
   WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
fi

if [ $RANK -eq 0 ]; then

ray start --head --port 6495 --num-gpus 8 --metrics-export-port 9099 \
   --node-ip-address 127.0.0.1 --disable-usage-stats \
   --temp-dir /tmp/ray/claw95-rti-8gpu

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
   ${CUSTOM_ARGS[@]} \
   ${INSPECTOR_ARGS[@]}

sleep 30m

else

echo "Starting Ray worker to ${MASTER_ADDR}"
ray start --block --address ${MASTER_ADDR}:6495 --num-gpus 8 --metrics-export-port 9099 \
   --disable-usage-stats --temp-dir /tmp/ray/claw95-rti-8gpu

fi
