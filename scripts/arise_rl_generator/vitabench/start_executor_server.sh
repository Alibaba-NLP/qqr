#!/bin/bash

# =============================================================================
# Start the executor agent inference service used for solver validation
#
# This service must be running before training the generator curriculum agent.
# The executor agent runs Qwen3.5-9B and validates generated tasks in interactive mode.
#
# Default: this service uses GPU 0 and training uses the remaining GPUs
# =============================================================================

set -ex

# Executor model path (Qwen3.5-9B)
EXECUTOR_MODEL_PATH=${EXECUTOR_MODEL_PATH:-"Qwen/Qwen3.5-9B"}

# Service port
EXECUTOR_PORT=${EXECUTOR_PORT:-30000}

# GPU settings
EXECUTOR_GPUS=${EXECUTOR_GPUS:-"0"}

# Tensor-parallel size
EXECUTOR_TP_SIZE=${EXECUTOR_TP_SIZE:-1}

echo "======================================"
echo "Starting the executor agent inference service (solver validation)"
echo "Model: Qwen3.5-9B"
echo "Path: ${EXECUTOR_MODEL_PATH}"
echo "Port: ${EXECUTOR_PORT}"
echo "GPU: ${EXECUTOR_GPUS}"
echo "TP Size: ${EXECUTOR_TP_SIZE}"
echo "======================================"

CUDA_VISIBLE_DEVICES=${EXECUTOR_GPUS} python -m sglang.launch_server \
    --model-path ${EXECUTOR_MODEL_PATH} \
    --port ${EXECUTOR_PORT} \
    --host 0.0.0.0 \
    --tensor-parallel-size ${EXECUTOR_TP_SIZE} \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --trust-remote-code \
    --tool-call-parser qwen
