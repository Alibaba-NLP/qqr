#!/bin/bash
# Usage: bash scripts/arise_rl_generator/travel/run-qwen3.5-9B.sh 2>&1 | tee travel_generator_$(date +%m%d).log
# =============================================================================
# Travel curriculum agent (generator) training script - Qwen3.5-9B + RG-KL
#
# The generator calls the MCP tools (AMap/Transport/WebSearch) to obtain real data, then
# generates query + expected_tools + rubrics.
#
# RG-KL algorithm:
#   Phase A: student rollout without coach memory -> generate tasks -> solver validation
#   Phase B: the gpt-5.2 coach reviews task quality -> produces coaching memory
#   Phase C: teacher rollout with coach memory injected, used only to estimate Δ_r
# =============================================================================

# =============================================================================
# Clean up any previous training processes
# Note: do not `pkill -9 sglang`; it would kill the external executor server (the solver on GPU 0)
# Only clean up ray and train.py; sglang processes are torn down indirectly by ray stop, since the training engine is a ray actor
# =============================================================================
ray stop --force 2>/dev/null || true
sleep 3
pkill -9 -f "ray::" 2>/dev/null || true
sleep 2
pkill -9 -f "train\.py" 2>/dev/null || true
pkill -9 -f "slime" 2>/dev/null || true
sleep 3
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "train\.py" 2>/dev/null || true
rm -rf /tmp/ray/*

set -ex

# =============================================================================
# Network settings
# =============================================================================

# =============================================================================
# Environment settings
# =============================================================================

RUN_NAME=${RUN_NAME:-Qwen3.5-9B-ARISE-Generator-Travel}
CKPT_DIR=${RUN_NAME}

QQR_PATH=$(pip list | grep qqr | awk '{print $NF}')
SLIME_PATH=$(pip list | grep slime | awk '{print $NF}')
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')

if [ -z "${QQR_PATH}" ]; then
  echo "QQR_PATH is not set"
  exit 1
fi

if [ -z "${SLIME_PATH}" ]; then
  echo "SLIME_PATH is not set"
  exit 1
fi

cd ${SLIME_PATH}

# =============================================================================
# GPU settings
# GPU 0 serves the executor; GPUs 1-7 are used for training
# =============================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2,3,4,5,6,7"}    # OOM mitigation: with 7 GPUs at TP=1 the logits blow up memory; use 6 GPUs at TP=2 and leave GPU 1 free

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23456}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-6}    # OOM mitigation: 6 GPUs with TP=2, DP=3
export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}

export PYTHONBUFFERED=16
export PYTHONPATH=${QQR_PATH}:${SLIME_PATH}:${MEGATRON_LM_PATH}:${PYTHONPATH}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)

if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_NVLS_ENABLE=${HAS_NVLINK}
export NCCL_DEBUG=WARN
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export TORCH_USE_CUDA_DSA=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export RAY_NUM_SERVER_CALL_THREAD=1
export TOKENIZERS_PARALLELISM=false
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True was tried, but slime's
# torch_memory_saver does not support expandable_segments and the sglang engine crashes on startup.
# The default allocator must be used, with max-tokens-per-gpu lowered to control memory.

# =============================================================================
# Ray memory management settings
# =============================================================================

export RAY_memory_monitor_refresh_ms=0
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/tmp/ray_spill"}}'
mkdir -p /tmp/ray_spill
export RAY_object_store_memory=$((200 * 1024 * 1024 * 1024))
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_PLASMA_STORE_MEMORY_FRACTION=0.3

adjust_oom_score() {
    echo -500 > /proc/self/oom_score_adj 2>/dev/null || true
}
adjust_oom_score

# =============================================================================
# Executor agent settings (solver validation; must be started beforehand)
# =============================================================================

export EXECUTOR_API_BASE=${EXECUTOR_API_BASE:-"http://localhost:30000/v1"}
export EXECUTOR_MODEL=${EXECUTOR_MODEL:-"default"}
export EXECUTOR_NUM_TRIALS=${EXECUTOR_NUM_TRIALS:-8}   # K=8, as in the paper
export EXECUTOR_SUCCESS_THRESHOLD=${EXECUTOR_SUCCESS_THRESHOLD:-0.9}  # γ=0.9, as in the paper
export EXECUTOR_MAX_STEPS=${EXECUTOR_MAX_STEPS:-40}    # the solver gets at most 40 rounds of tool interaction, as in the paper
export EXECUTOR_CONCURRENCY_LIMIT=${EXECUTOR_CONCURRENCY_LIMIT:-8}

# Coach (gpt-5.2) produces the generator coaching memory
export COACH_MODEL="gpt-5.2-2025-12-11"
export COACH_CONCURRENCY_LIMIT="10"

# =============================================================================
# Reward-Gated Reverse KL (RG-KL) settings
# =============================================================================

export ENABLE_RG_KL=true
export K_STUDENT=8                          # number of student rollouts (these are trained on)
export K_TEACHER=8                          # number of teacher rollouts (used only for Δ_r); G = 8 + 8 = 16, as in the paper
export RG_KL_LAMBDA_0=0.5                   # λ_0=0.5, as in the paper
export RG_KL_DELTA_THRESHOLD=0.05           # τ=0.05, as in the paper
export RG_KL_GATE_TEMPERATURE=0.0125        # sigmoid gate temperature T
export RG_KL_WARMUP_ITERS=20                # cosine warm-up
export RG_KL_DECAY_ITERS=120                # cosine decay length
export RG_KL_MIN_LAMBDA_RATIO=0.1
export ENABLE_GUIDED_TOKENS_FOR_STUDENT=true

# =============================================================================
# API Keys
# =============================================================================

export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:?"DASHSCOPE_API_KEY is required"}
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export AMAP_MAPS_API_KEY=${AMAP_MAPS_API_KEY:?"AMAP_MAPS_API_KEY is required (AMap open platform)"}
export BAILIAN_WEB_SEARCH_API_KEY=${BAILIAN_WEB_SEARCH_API_KEY:-}
export SEARCH_API_KEY=${SEARCH_API_KEY:-}
export SEARCH_API_URL=${SEARCH_API_URL:-}

# =============================================================================
# Model settings (Qwen3.5-9B)
# =============================================================================

source "${QQR_PATH}/scripts/models/qwen3.5-9B.sh"

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3.5-9B
   --ref-load /path/to/Qwen3.5-9B_torch_dist
   --load ${CKPT_DIR}
   --save ${CKPT_DIR}
   --save-interval 10
   --normalize-advantages
)

# =============================================================================
# Rollout settings
# =============================================================================

ROLLOUT_ARGS=(
   --rollout-function-path qqr.rollout.agent_rollout.generate_rollout
   --prompt-data ${TRAIN_DATA:?"TRAIN_DATA is required (task seed jsonl, one query per line)"}
   --input-key query
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 3
   --n-samples-per-prompt 16              # G=16 (8 student + 8 teacher), as in the paper
   --rollout-max-context-len 32000
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 48                 # rollout_batch_size × 16
   --balance-data
   --group-rm
)

# =============================================================================
# Evaluation settings
# =============================================================================

EVAL_ARGS=(
   --skip-eval-before-train                # Generator training does not need a baseline evaluation (each one takes about 4h)
   --eval-interval 10
   --eval-prompt-data ${QQR_PATH}/data/ecr_travel/test.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-context-len 32000
   --eval-input-key query
   --eval-temperature 0.0
)

# =============================================================================
# Performance settings
# =============================================================================

PERF_ARGS=(
   --tensor-model-parallel-size 2          # OOM mitigation: at TP=1 the logits need 20GB+ on a single GPU; TP=2 splits them across 2 GPUs, halving per-GPU memory
   --sequence-parallel                     # Enable sequence parallelism when TP>1 to further reduce activation memory
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096               # With TP=2 the activation is halved, so this can go back to 4096
)

# =============================================================================
# RG-KL GRPO settings
# =============================================================================

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-rg-kl
   --rg-kl-clip-value 10.0
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.2
   --clip-grad 100000
)

# =============================================================================
# Optimizer settings
# =============================================================================

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --min-lr 1e-7
   --lr-decay-style cosine
   --lr-decay-iters 480
   --lr-warmup-iters 0
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --override-opt_param-scheduler
)

# =============================================================================
# Logging settings
# =============================================================================


SWANLAB_ARGS=(
   # --use-swanlab
   # --swanlab-project QQR-TravelCurriculum-RG-KL
   # --swanlab-group ${RUN_NAME}
   # --swanlab-mode cloud
)

# =============================================================================
# SGLang inference engine settings
# =============================================================================

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2         # TP=2 means 2 GPUs per engine (3 engines total), matching the training TP
   --sglang-mem-fraction-static 0.4        # OOM mitigation: 0.5 -> 0.4 so sglang leaves more room for Megatron
   --sglang-server-concurrency 64          # combined with 0.4 to further reduce total KV cache
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# =============================================================================
# Custom function paths (pointing at arise_rl_generator.travel)
# =============================================================================

CUSTOM_ARGS=(
   --custom-generate-function-path qqr.examples.arise_rl_generator.travel.generate
   --custom-rm-path qqr.examples.arise_rl_generator.travel.group_reward
   --custom-reward-post-process-path qqr.examples.arise_rl_generator.travel.reward_post_process
)

# =============================================================================
# Launch training
# =============================================================================

if [ $RANK -eq 0 ]; then

echo "Checking the executor solver service..."
if curl -s --max-time 5 ${EXECUTOR_API_BASE}/models > /dev/null 2>&1; then
    echo "Executor service is ready: ${EXECUTOR_API_BASE}"
else
    echo "Warning: the executor service did not respond (${EXECUTOR_API_BASE})"
    echo "  Make sure the executor service is running:"
    echo "  bash scripts/arise_rl_generator/travel/start_executor_server.sh"
    sleep 10
fi

ray start --head --port 6379 --disable-usage-stats --metrics-export-port 8080

mkdir -p ${CKPT_DIR}/debug_generator_travel
mkdir -p ${CKPT_DIR}/memory

echo "======================================"
echo "Travel curriculum RG-KL training configuration:"
echo "  Model: Qwen3.5-9B"
echo "  Algorithm: Reward-Gated Reverse KL"
echo "  GPUs: ${NPROC_PER_NODE} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "  Executor: ${EXECUTOR_API_BASE}"
echo "  Coach: ${COACH_MODEL}"
echo "  RG-KL: k_s=${K_STUDENT} k_m=${K_TEACHER} λ_0=${RG_KL_LAMBDA_0}"
echo "======================================"

python3 train.py \
   --actor-num-nodes ${NNODES} \
   --actor-num-gpus-per-node ${NPROC_PER_NODE} \
   --rollout-num-gpus ${NPROC_PER_NODE} \
   --colocate \
   --save-rollout-data \
   --save-debug-rollout-data "${CKPT_DIR}/debug_generator_travel/rollout_{rollout_id}.pt" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${SWANLAB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   2>&1 | tee ${CKPT_DIR}/train.log

sleep 30m

else

echo "Starting Ray worker to ${MASTER_ADDR}"
ray start --block --address ${MASTER_ADDR}:6379 --disable-usage-stats --metrics-export-port 8080

fi
