#!/bin/bash
# Usage: bash scripts/arise_rl_solver/vitabench/run-qwen3.5-9B.sh 2>&1 | tee vita_solver_$(date +%m%d).log
# =============================================================================
# VitaBench solver training script - Reward-Gated Reverse KL (RG-KL)
#
# Core idea (RG-SED applied to VitaBench interactive multi-turn settings):
# - Trains the agent in interactive mode against the VitaBench simulated environment (OTA domain)
# - UserSimulator (local Qwen3.5-397B-A17B) reveals user requirements over multiple turns
# - The agent interacts with the user and calls tools to complete OTA tasks
# - Reward = 0.8 · rubric_rate + 0.2 · 𝟙[all satisfied], scored by a sliding-window LLM judge
#
# RG-KL algorithm:
#   Phase A: student rollout without memory -> rubric scoring -> cache
#   Phase B: the LLM coach (gpt-5.2) produces query-specific solving advice (memory)
#   Phase C: teacher rollout with memory injected, used only to estimate Δ_r (reward neutralised, loss_mask=0)
#   Training:
#     - Student samples use guided_tokens = [p_m_prompt, τ_s_response]
#       → slime's forward pass yields teacher_log_probs = π_θ(τ_s | p_m)
#       → apply_rg_kl_to_advantages: adv -= λ(Δ_r) · clamp(student_lp - teacher_lp)
#     - L = L_GRPO_student + λ(Δ_r) · D_KL(π_θ(·|p_s) ‖ π_θ(·|p_m)) + β · KL_ref_penalty
#     - λ(Δ_r) = λ_0 · gate(Δ_r) · warmup · cosine_decay
#
# Differences from the VitaBench memory-guided off-policy GRPO variant:
#   - Drop --use-guided-tokens-for-training: no more p_s token-swap importance sampling on the teacher
#   - Drop --use-rollout-logprobs: back to standard on-policy instead of off-policy importance sampling
#   + Add --use-rg-kl to enable reward-gated reverse KL
#   + kl-loss-coef 0.005 -> 0.001 (beta): safety anchor, kept small so RG-KL dominates
# =============================================================================

# =============================================================================
# Clean up any previous training processes
# =============================================================================
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

# =============================================================================
# Network settings
# =============================================================================

# =============================================================================
# Environment settings
# =============================================================================

RUN_NAME=${RUN_NAME:-Qwen3.5-9B-ARISE-Solver-VitaBench}
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

# =============================================================================
# VitaBench settings
# =============================================================================

# scenario: ota (online travel) / delivery / instore / cross_domain
# the generator and solver prompts switch to the matching scenario automatically based on this value
export VITABENCH_DOMAIN=${VITABENCH_DOMAIN:-ota}
export VITABENCH_DATA_DIR=${VITABENCH_DATA_DIR:?"VITABENCH_DATA_DIR is required (the data/vita directory of the official VitaBench repo)"}

# UserSimulator (locally deployed Qwen3-397B-A17B, no rate limit)
export USER_SIMULATOR_API_KEY="EMPTY"
export USER_SIMULATOR_BASE_URL=${USER_SIMULATOR_BASE_URL:?"USER_SIMULATOR_BASE_URL is required (Qwen3.5-397B service endpoint)"}
export USER_SIMULATOR_MODEL="Qwen3.5-397B-A17B"
export USER_SIMULATOR_TEMPERATURE="0.0"

# Coach (external gpt-5.2) produces query-specific memory
export COACH_MODEL="gpt-5.2-2025-12-11"
export COACH_CONCURRENCY_LIMIT="10"

# Sliding-window judge (local Qwen3.5-397B)
export LLM_JUDGE_CONCURRENCY_LIMIT="16"

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
export RG_KL_MIN_LAMBDA_RATIO=0.1           # decays to λ_0 · 0.1 at the end
export ENABLE_GUIDED_TOKENS_FOR_STUDENT=true  # whether to build [p_m, τ_s] guided_tokens for the student

export ENABLE_MEMORY_GUIDED_OFFPOLICY=false

# =============================================================================
# Model settings
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
   --prompt-data ${TRAIN_DATA:?"TRAIN_DATA is required (interactive training jsonl, convertible from the official tasks.json or produced by the generator)"}
   --input-key query
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 2
   --n-samples-per-prompt 16              # G=16 (8 student + 8 teacher), as in the paper
   --rollout-max-context-len 32000
   --rollout-max-response-len 10000
   --rollout-temperature 0.8

   --global-batch-size 32                 # rollout_batch_size × 16
   --balance-data
   --group-rm
)

# =============================================================================
# Evaluation settings
# =============================================================================

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data ${EVAL_DATA:?"EVAL_DATA is required (evaluation jsonl, convertible from the official tasks.json)"}
   --n-samples-per-eval-prompt 1
   --eval-max-context-len 32000
   --eval-input-key query
   --eval-temperature 0.0
)

# =============================================================================
# Performance settings
# =============================================================================

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
   --max-tokens-per-gpu 20480              # Multi-turn sequences plus memory are long, so keep this low to avoid OOM
)

# =============================================================================
# RG-KL GRPO settings
# =============================================================================
# Key differences from the memory-guided off-policy variant:
#   - Drop --use-guided-tokens-for-training: no more p_s swap on the teacher
#   - Drop --use-rollout-logprobs: on-policy training needs no importance-sampling ratio
#   + Add --use-rg-kl and --rg-kl-clip-value 10.0
#   + --kl-loss-coef 0.005 -> 0.001 so RG-KL dominates and the reference KL acts only as a safety anchor

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-rg-kl                             # * Enable Reward-Gated Reverse KL
   --rg-kl-clip-value 10.0                 # per-token KL clipping threshold
   --use-kl-loss
   --kl-loss-coef 0.001                    # Reference-KL safety anchor (far below the RG-KL peak coefficient of about 0.087)
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.2                     # symmetric clip; on-policy training does not need it loosened
   --clip-grad 100000                      # VitaBench multi-turn sequences produce large grad_norm; raise the clip so the effective LR does not collapse
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
   # --swanlab-project QQR-VitaBench-RG-KL-v2
   # --swanlab-group ${RUN_NAME}
   # --swanlab-mode cloud
)

# =============================================================================
# SGLang inference engine settings
# =============================================================================

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 512
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# =============================================================================
# Custom function paths (pointing at the arise_rl_solver.vitabench module)
# =============================================================================

CUSTOM_ARGS=(
   --custom-generate-function-path qqr.examples.arise_rl_solver.vitabench.generate
   --custom-rm-path qqr.examples.arise_rl_solver.vitabench.group_reward
   --custom-reward-post-process-path qqr.examples.arise_rl_solver.vitabench.reward_post_process
)

# =============================================================================
# API Keys
# =============================================================================

export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:?"DASHSCOPE_API_KEY is required"}
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# =============================================================================
# Launch training
# =============================================================================

if [ $RANK -eq 0 ]; then

ray start --head --port 6379 --disable-usage-stats --metrics-export-port 8080

mkdir -p ${CKPT_DIR}/debug_solver_vitabench
python3 train.py \
   --actor-num-nodes ${NNODES} \
   --actor-num-gpus-per-node ${NPROC_PER_NODE} \
   --rollout-num-gpus ${NPROC_PER_NODE} \
   --colocate \
   --save-rollout-data \
   --save-debug-rollout-data "${CKPT_DIR}/debug_solver_vitabench/rollout_{rollout_id}.pt" \
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
