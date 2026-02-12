#!/bin/bash
set -euo pipefail

# Single-GPU SM scavenging test
# Uses GPU 1 (adjust if needed)
GPU_ID=1
VLLM_PORT=8100
DURATION=60
GEN_CONCURRENCY=8
GEN_MAX_TOKENS=256
EMBED_BATCH_SIZE=32
EMBED_SEQ_LENGTH=128
MODEL="Qwen/Qwen2.5-3B-Instruct"

DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$DIR/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== MPS Co-location Experiment ==="
echo "GPU: $GPU_ID"
echo "Generative model: $MODEL"
echo "Embedding model: intfloat/e5-large-v2"
echo "Results: $RESULTS_DIR"
echo ""

cleanup() {
    echo "Cleaning up..."
    # Kill vLLM if running
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    # Kill embedding benchmark if running
    if [ -n "${EMBED_PID:-}" ]; then
        kill "$EMBED_PID" 2>/dev/null || true
        wait "$EMBED_PID" 2>/dev/null || true
    fi
    # Stop MPS if running (use our experiment-specific pipe)
    echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-experiment-$GPU_ID nvidia-cuda-mps-control 2>/dev/null || true
}
trap cleanup EXIT

wait_for_vllm() {
    echo "Waiting for vLLM to be ready..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "vLLM ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: vLLM failed to start after 120s"
    return 1
}

start_vllm() {
    echo "Starting vLLM on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$VLLM_PORT" \
        --gpu-memory-utilization 0.7 \
        --max-model-len 2048 \
        --disable-log-requests \
        > "$RESULTS_DIR/vllm_${1}.log" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm
}

stop_vllm() {
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        unset VLLM_PID
        sleep 2
    fi
}

record_gpu_mem() {
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits -i "$GPU_ID" \
        | tee -a "$RESULTS_DIR/gpu_mem_${1}.txt"
}

# ============================================================
# STEP 1: Generative-only baseline
# ============================================================
echo "=========================================="
echo "STEP 1: Generative-only baseline"
echo "=========================================="

start_vllm "step1"
echo "GPU memory after vLLM load:"
record_gpu_mem "step1_after_load"

python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    2>&1 | tee "$RESULTS_DIR/step1_gen.txt"

echo "GPU memory during decode:"
record_gpu_mem "step1_during_decode"

stop_vllm

echo ""
echo "Step 1 complete."
echo ""

# ============================================================
# STEP 2: Co-located with MPS
# ============================================================
echo "=========================================="
echo "STEP 2: Co-located with MPS"
echo "=========================================="

# Start MPS daemon for GPU
echo "Starting MPS daemon..."
MPS_PIPE=/tmp/nvidia-mps-experiment-$GPU_ID
MPS_LOG=/tmp/nvidia-mps-experiment-log-$GPU_ID
mkdir -p "$MPS_PIPE" "$MPS_LOG"
# Start daemon with CUDA_VISIBLE_DEVICES so it binds to the right physical GPU
CUDA_VISIBLE_DEVICES=$GPU_ID \
CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE \
CUDA_MPS_LOG_DIRECTORY=$MPS_LOG \
nvidia-cuda-mps-control -d
sleep 2
echo "MPS daemon started."

# Client processes: set CUDA_VISIBLE_DEVICES=0 (the MPS daemon exposes 1 GPU as device 0)
# and point to the MPS pipe directory.
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE

# Start vLLM under MPS
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_step2.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

echo "GPU memory after vLLM load (MPS):"
record_gpu_mem "step2_after_load"

# Start embedding benchmark under MPS (in background)
python3 "$DIR/embed_benchmark.py" \
    --duration "$DURATION" \
    --batch-size "$EMBED_BATCH_SIZE" \
    --seq-length "$EMBED_SEQ_LENGTH" \
    > "$RESULTS_DIR/step2_embed.txt" 2>&1 &
EMBED_PID=$!

# Give embedding model a moment to load
sleep 10
echo "GPU memory with both models (MPS):"
record_gpu_mem "step2_both_loaded"

# Run generative benchmark concurrently
python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    2>&1 | tee "$RESULTS_DIR/step2_gen.txt"

# Wait for embedding to finish
wait "$EMBED_PID" 2>/dev/null || true
unset EMBED_PID
echo "Embedding results:"
cat "$RESULTS_DIR/step2_embed.txt"

stop_vllm

# Stop MPS and clean up env
echo "Stopping MPS daemon..."
echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE nvidia-cuda-mps-control 2>/dev/null || true
unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_VISIBLE_DEVICES
sleep 2

echo ""
echo "Step 2 complete."
echo ""

# ============================================================
# STEP 3: Co-located without MPS (time-sliced)
# ============================================================
echo "=========================================="
echo "STEP 3: Co-located without MPS (time-sliced)"
echo "=========================================="

start_vllm "step3"
echo "GPU memory after vLLM load (no MPS):"
record_gpu_mem "step3_after_load"

# Start embedding benchmark (no MPS, same GPU)
CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$DIR/embed_benchmark.py" \
    --duration "$DURATION" \
    --batch-size "$EMBED_BATCH_SIZE" \
    --seq-length "$EMBED_SEQ_LENGTH" \
    > "$RESULTS_DIR/step3_embed.txt" 2>&1 &
EMBED_PID=$!

sleep 10
echo "GPU memory with both models (no MPS):"
record_gpu_mem "step3_both_loaded"

# Run generative benchmark concurrently
python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    2>&1 | tee "$RESULTS_DIR/step3_gen.txt"

wait "$EMBED_PID" 2>/dev/null || true
unset EMBED_PID
echo "Embedding results:"
cat "$RESULTS_DIR/step3_embed.txt"

stop_vllm

echo ""
echo "Step 3 complete."
echo ""

# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Step 1 (gen only):"
grep "Throughput:" "$RESULTS_DIR/step1_gen.txt" || true
echo ""
echo "Step 2 (MPS):"
echo "  Gen:"
grep "Throughput:" "$RESULTS_DIR/step2_gen.txt" || true
echo "  Embed:"
grep "Throughput:" "$RESULTS_DIR/step2_embed.txt" || true
echo ""
echo "Step 3 (time-sliced):"
echo "  Gen:"
grep "Throughput:" "$RESULTS_DIR/step3_gen.txt" || true
echo "  Embed:"
grep "Throughput:" "$RESULTS_DIR/step3_embed.txt" || true
echo ""
echo "Full results in: $RESULTS_DIR"
