#!/bin/bash
# Usage: bash scripts/arise_rl_generator/deepresearch/run-qwen3.5-9B.sh 2>&1 | tee dr_generator_$(date +%m%d).log
# =============================================================================
# DeepResearch curriculum agent (generator) training script - Qwen3.5-9B + RG-KL
#
# The generator calls web_search (Google Search MCP) to obtain real data, then
# generates query + rubrics. The solver (executor service) answers and the rubric pass rate is scored.
# =============================================================================

# Note: do not `pkill -9 sglang`; it would kill the external executor server (the solver on GPU 0)
# Only clean up ray and train.py; the sglang training engine is a ray actor and is torn down by ray stop
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


RUN_NAME=${RUN_NAME:-Qwen3.5-9B-ARISE-Generator-DeepResearch}
CKPT_DIR=${RUN_NAME}

QQR_PATH=$(pip list | grep qqr | awk '{print $NF}')
SLIME_PATH=$(pip list | grep slime | awk '{print $NF}')
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')

cd ${SLIME_PATH}

# GPU 0 serves the executor (solver); GPUs 1-7 train the generator
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4,5,6,7"}

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23456}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-7}
export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}

export PYTHONBUFFERED=16
export PYTHONPATH=${QQR_PATH}:${SLIME_PATH}:${MEGATRON_LM_PATH}:${PYTHONPATH}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

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

# Ray memory management
export RAY_memory_monitor_refresh_ms=0
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/tmp/ray_spill"}}'
mkdir -p /tmp/ray_spill
export RAY_object_store_memory=$((200 * 1024 * 1024 * 1024))
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_PLASMA_STORE_MEMORY_FRACTION=0.3
adjust_oom_score() { echo -500 > /proc/self/oom_score_adj 2>/dev/null || true; }
adjust_oom_score

# Executor solver service (must be started beforehand)
export EXECUTOR_API_BASE=${EXECUTOR_API_BASE:-"http://localhost:30000/v1"}
export EXECUTOR_MODEL=${EXECUTOR_MODEL:-"default"}
export EXECUTOR_NUM_TRIALS=${EXECUTOR_NUM_TRIALS:-8}   # K=8, as in the paper
export EXECUTOR_SUCCESS_THRESHOLD=${EXECUTOR_SUCCESS_THRESHOLD:-0.9}  # γ=0.9, as in the paper
export EXECUTOR_MAX_STEPS=${EXECUTOR_MAX_STEPS:-40}    # the solver gets at most 40 rounds of tool interaction, as in the paper
export EXECUTOR_CONCURRENCY_LIMIT=${EXECUTOR_CONCURRENCY_LIMIT:-8}

# Coach
export COACH_MODEL="gpt-5.2-2025-12-11"

# RG-KL
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

# API Keys
export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:?"DASHSCOPE_API_KEY is required"}
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export SEARCH_API_KEY=${SEARCH_API_KEY:-}
export SEARCH_API_URL=${SEARCH_API_URL:-}
export SERPER_API_KEY="${SERPER_API_KEY:-}"
export SEARCH_BACKEND="${SEARCH_BACKEND:-google}"

# Model
source "${QQR_PATH}/scripts/models/qwen3.5-9B.sh"

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3.5-9B
   --ref-load /path/to/Qwen3.5-9B_torch_dist
   --load ${CKPT_DIR}
   --save ${CKPT_DIR}
   --save-interval 10
   --normalize-advantages
)

# Use the deepresearch training data as prompt seeds; the generator creates tasks from random topics
# Task seed data: one query per line, each triggering a generation
TRAIN_DATA=${TRAIN_DATA:?"TRAIN_DATA is required (task seed jsonl)"}
EVAL_DATA=${EVAL_DATA:?"EVAL_DATA is required (evaluation jsonl, e.g. data/ecr_deepresearch/test.jsonl)"}

ROLLOUT_ARGS=(
   --rollout-function-path qqr.rollout.agent_rollout.generate_rollout
   --prompt-data ${TRAIN_DATA}
   --input-key query
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 2
   --n-samples-per-prompt 16              # G=16 (8 student + 8 teacher), as in the paper
   --rollout-max-context-len 32000
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   --global-batch-size 32                 # rollout_batch_size × 16
   --balance-data
   --group-rm
)

EVAL_ARGS=(
   --skip-eval-before-train                # Generator training does not need a baseline evaluation
   --eval-interval 10
   --eval-prompt-data ${EVAL_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-context-len 32000
   --eval-input-key query
   --eval-temperature 0.0
)

PERF_ARGS=(
   --tensor-model-parallel-size 1          # 7 GPUs are not divisible by 4, so use TP=1 (DP=7)
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096               # OOM mitigation: 20480 -> 4096
)

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


SWANLAB_ARGS=(
   # --use-swanlab
   # --swanlab-project QQR-DeepResearchCurriculum-RG-KL
   # --swanlab-group ${RUN_NAME}
   # --swanlab-mode cloud
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1         # 7 GPUs are not divisible by 2, so use one GPU per engine (7 engines total)
   --sglang-mem-fraction-static 0.5        # OOM mitigation: 0.7 -> 0.5 to leave more memory for Megatron training
   --sglang-server-concurrency 128
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path qqr.examples.arise_rl_generator.deepresearch.generate
   --custom-rm-path qqr.examples.arise_rl_generator.deepresearch.group_reward
   --custom-reward-post-process-path qqr.examples.arise_rl_generator.deepresearch.reward_post_process
)

if [ $RANK -eq 0 ]; then

echo "Checking the executor solver service..."
if curl -s --max-time 5 ${EXECUTOR_API_BASE}/models > /dev/null 2>&1; then
    echo "Executor service is ready: ${EXECUTOR_API_BASE}"
else
    echo "Warning: the executor service did not respond (${EXECUTOR_API_BASE})"
    sleep 10
fi

ray start --head --port 6379 --disable-usage-stats --metrics-export-port 8080

mkdir -p ${CKPT_DIR}/debug_generator_deepresearch
mkdir -p ${CKPT_DIR}/memory

echo "======================================"
echo "DeepResearch Curriculum RG-KL:"
echo "  Model: Qwen3.5-9B"
echo "  Algorithm: Reward-Gated Reverse KL"
echo "  RG-KL: k_s=${K_STUDENT} k_m=${K_TEACHER} λ_0=${RG_KL_LAMBDA_0}"
echo "======================================"

python3 train.py \
   --actor-num-nodes ${NNODES} \
   --actor-num-gpus-per-node ${NPROC_PER_NODE} \
   --rollout-num-gpus ${NPROC_PER_NODE} \
   --colocate \
   --save-rollout-data \
   --save-debug-rollout-data "${CKPT_DIR}/debug_generator_deepresearch/rollout_{rollout_id}.pt" \
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

ray start --block --address ${MASTER_ADDR}:6379 --disable-usage-stats --metrics-export-port 8080

fi
