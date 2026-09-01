#!/bin/bash
# ============================================================
# Travel rubrics solver training script (Qwen3.5-9B)
# Reward-Gated Reverse KL (RG-KL)
#   - Three-phase rollout: student (k_s=8) + teacher (k_m=8); teacher is used only to estimate Δ_r
#   - L = L_GRPO_student + λ(Δ_r) · KL(π_θ(·|p_s) ‖ π_θ(·|p_m)) + β · KL_ref
#   - λ(Δ_r) = 0.1 · max(Δ_r - 0.05, 0) · warmup · cosine_decay
# Usage: bash scripts/arise_rl_solver/travel/run-qwen3.5-9B.sh 2>&1 | tee travel_solver_$(date +%m%d).log
# ============================================================

# Clean up any previous training processes
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f "train\.py" 2>/dev/null || true
pkill -9 -f "slime" 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 -f "train\.py" 2>/dev/null || true
rm -rf /tmp/ray/*

set -ex

RUN_NAME=${RUN_NAME:-Qwen3.5-9B-ARISE-Solver-Travel}
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

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23456}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
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

source "${QQR_PATH}/scripts/models/qwen3.5-9B.sh"

# -- LLM judge settings --
export LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL:-"gpt-5.2-2025-12-11"}
export LLM_JUDGE_CONCURRENCY_LIMIT=${LLM_JUDGE_CONCURRENCY_LIMIT:-10}

# -- Reward-Gated Reverse KL (RG-KL) settings --
export ENABLE_RG_KL=true
export K_STUDENT=8                          # number of student rollouts (these are trained on)
export K_TEACHER=8                          # number of teacher rollouts (used only for Δ_r); G = 8 + 8 = 16, as in the paper
export RG_KL_LAMBDA_0=0.5                   # λ_0=0.5, as in the paper
export RG_KL_DELTA_THRESHOLD=0.05           # τ=0.05, as in the paper
export RG_KL_GATE_TEMPERATURE=0.0125        # sigmoid gate temperature T
export RG_KL_WARMUP_ITERS=20                # cosine warm-up
export RG_KL_DECAY_ITERS=120                # cosine decay length
export RG_KL_MIN_LAMBDA_RATIO=0.1           # decays to λ_0 · 0.1 at the end
export KL_SCOPE=final_only                  # * Switched from all_response to final_only so tool tokens are excluded from the KL
export ENABLE_GUIDED_TOKENS_FOR_STUDENT=true  # whether to build [p_m, τ_s] guided_tokens for the student

# Disable the older memory-guided off-policy path to avoid conflicts
export ENABLE_MEMORY_GUIDED_OFFPOLICY=false

# -- Data paths --
# Training data (query+expected_tools+rubrics jsonl; can be produced by the arise_rl_generator/travel generator)
TRAIN_DATA=${TRAIN_DATA:?"TRAIN_DATA is required (query+expected_tools+rubrics jsonl)"}
EVAL_DATA=${QQR_PATH}/data/ecr_travel/test.jsonl

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3.5-9B
   --ref-load /path/to/Qwen3.5-9B_torch_dist
   --load ${CKPT_DIR}
   --save ${CKPT_DIR}
   --save-interval 50
   --normalize-advantages
)

ROLLOUT_ARGS=(
   --rollout-function-path qqr.rollout.agent_rollout.generate_rollout
   --prompt-data ${TRAIN_DATA}
   --input-key query
   --rollout-shuffle
   --num-rollout 120                      # train on everything
   --rollout-batch-size 4
   --n-samples-per-prompt 16              # G=16 (8 student + 8 teacher), as in the paper
   --rollout-max-context-len 32000
   --rollout-max-response-len 10000
   --rollout-temperature 0.8

   --global-batch-size 64                 # rollout_batch_size × 16
   --balance-data
   --group-rm
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data ${EVAL_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-context-len 32000
   --eval-input-key query
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
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

# -- RG-KL loss settings --
# Differences from the memory-guided off-policy variant:
#   - Drop --use-guided-tokens-for-training: no more token-swap importance sampling
#   - Drop --use-rollout-logprobs: back to standard on-policy instead of off-policy importance sampling
#   + Add --use-rg-kl to enable reward-gated reverse KL
#   + --kl-loss-coef 0.001 (beta): reference-KL safety anchor lowered from 0.005 so RG-KL dominates
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-rg-kl                           # * Enable Reward-Gated Reverse KL
   --rg-kl-clip-value 10.0               # per-token KL clipping threshold
   --use-kl-loss
   --kl-loss-coef 0.001                  # Reference-KL safety anchor (far below the RG-KL λ_0 = 0.5)
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.2                   # on-policy, so no need to loosen it
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --min-lr 1e-7
   --lr-decay-style cosine
   --lr-decay-iters 480                  # 120 rollout × 4 step = 480
   --lr-warmup-iters 0
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --override-opt_param-scheduler
)


SWANLAB_ARGS=(
   # --use-swanlab
   # --swanlab-project QQR-TravelRubrics-QWen3.5
   # --swanlab-group ${RUN_NAME}
   # --swanlab-mode cloud
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Custom function paths (pointing at the arise_rl_solver.travel module)
CUSTOM_ARGS=(
   --custom-generate-function-path qqr.examples.arise_rl_solver.travel.generate
   --custom-rm-path qqr.examples.arise_rl_solver.travel.group_reward
   --custom-reward-post-process-path qqr.examples.arise_rl_solver.travel.reward_post_process
)

export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:?"DASHSCOPE_API_KEY is required"}
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

export AMAP_MAPS_API_KEY=${AMAP_MAPS_API_KEY:?"AMAP_MAPS_API_KEY is required (AMap open platform)"}
export BAILIAN_WEB_SEARCH_API_KEY=${BAILIAN_WEB_SEARCH_API_KEY:-}


if [ $RANK -eq 0 ]; then

ray start --head --port 6379 --disable-usage-stats --metrics-export-port 8080

mkdir -p ${CKPT_DIR}/debug_solver_travel
python3 train.py \
   --actor-num-nodes ${NNODES} \
   --actor-num-gpus-per-node ${NPROC_PER_NODE} \
   --rollout-num-gpus ${NPROC_PER_NODE} \
   --colocate \
   --save-rollout-data \
   --save-debug-rollout-data "${CKPT_DIR}/debug_solver_travel/rollout_{rollout_id}.pt" \
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
