#!/bin/bash
set -euo pipefail

# Short nsys profiles for each scenario
# Uses GPU metrics sampling (system-wide) to see SM utilization and memory bandwidth

GPU_ID=1
VLLM_PORT=8100
MODEL="Qwen/Qwen2.5-3B-Instruct"
PROFILE_DURATION=15  # seconds of steady-state to capture
EMBED_BATCH_SIZE=32
EMBED_SEQ_LENGTH=128

DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$DIR/results/nsys_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== nsys Profiling ==="
echo "Results: $RESULTS_DIR"

VLLM_PID=""
EMBED_PID=""

cleanup() {
    echo "Cleaning up..."
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
    [ -n "${EMBED_PID:-}" ] && kill "$EMBED_PID" 2>/dev/null && wait "$EMBED_PID" 2>/dev/null || true
    echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-experiment-$GPU_ID nvidia-cuda-mps-control 2>/dev/null || true
}
trap cleanup EXIT

wait_for_vllm() {
    echo "Waiting for vLLM..."
    for i in $(seq 1 180); do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "vLLM ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: vLLM failed to start"
    return 1
}

stop_vllm() {
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 2
}

# ============================================================
# Profile 1: Generative-only
# ============================================================
echo ""
echo "=== Profile 1: Generative-only ==="

CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.7 --max-model-len 2048 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_p1.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

# Warm up
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" --duration 5 --concurrency 8 --max-tokens 64 > /dev/null 2>&1

# Start benchmark, then capture GPU metrics while it runs
python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration $((PROFILE_DURATION + 10)) \
    --concurrency 8 --max-tokens 256 \
    > "$RESULTS_DIR/p1_gen_output.txt" 2>&1 &
GEN_CLIENT_PID=$!

sleep 3  # let it reach steady state

echo "Capturing GPU metrics (gen only, ${PROFILE_DURATION}s)..."
# Use nvidia-smi dmon for per-GPU utilization sampling
nvidia-smi dmon -i $GPU_ID -s uct -d 1 -c $PROFILE_DURATION > "$RESULTS_DIR/dmon_gen_only.txt" 2>&1 &
DMON_PID=$!

wait "$DMON_PID" 2>/dev/null || true
wait "$GEN_CLIENT_PID" 2>/dev/null || true
echo "Profile 1 done."
echo "Gen throughput:"
grep "Throughput:" "$RESULTS_DIR/p1_gen_output.txt" || true
stop_vllm

# ============================================================
# Profile 2: Co-located with MPS
# ============================================================
echo ""
echo "=== Profile 2: Co-located with MPS ==="

MPS_PIPE=/tmp/nvidia-mps-experiment-$GPU_ID
MPS_LOG=/tmp/nvidia-mps-experiment-log-$GPU_ID
mkdir -p "$MPS_PIPE" "$MPS_LOG"
CUDA_VISIBLE_DEVICES=$GPU_ID CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" nvidia-cuda-mps-control -d
sleep 2
echo "MPS started."

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.7 --max-model-len 2048 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_p2.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

# Warm up gen
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" --duration 5 --concurrency 8 --max-tokens 64 > /dev/null 2>&1

# Start embedding
python3 "$DIR/embed_benchmark.py" --duration $((PROFILE_DURATION + 15)) --batch-size $EMBED_BATCH_SIZE --seq-length $EMBED_SEQ_LENGTH \
    > "$RESULTS_DIR/p2_embed_output.txt" 2>&1 &
EMBED_PID=$!
sleep 5  # let embedding warm up and reach steady state

# Start gen benchmark
python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration $((PROFILE_DURATION + 10)) \
    --concurrency 8 --max-tokens 256 \
    > "$RESULTS_DIR/p2_gen_output.txt" 2>&1 &
GEN_CLIENT_PID=$!

sleep 3

echo "Capturing GPU metrics (MPS, ${PROFILE_DURATION}s)..."
nvidia-smi dmon -i $GPU_ID -s uct -d 1 -c $PROFILE_DURATION > "$RESULTS_DIR/dmon_mps.txt" 2>&1 &
DMON_PID=$!

wait "$DMON_PID" 2>/dev/null || true
wait "$GEN_CLIENT_PID" 2>/dev/null || true
wait "$EMBED_PID" 2>/dev/null || true
EMBED_PID=""
echo "Profile 2 done."
echo "Gen throughput:"
grep "Throughput:" "$RESULTS_DIR/p2_gen_output.txt" || true
echo "Embed throughput:"
grep "Throughput:" "$RESULTS_DIR/p2_embed_output.txt" || true
stop_vllm

echo quit | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control 2>/dev/null || true
unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_VISIBLE_DEVICES
sleep 2

# ============================================================
# Profile 3: Co-located without MPS (time-sliced)
# ============================================================
echo ""
echo "=== Profile 3: Co-located without MPS ==="

CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.7 --max-model-len 2048 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_p3.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" --duration 5 --concurrency 8 --max-tokens 64 > /dev/null 2>&1

CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$DIR/embed_benchmark.py" \
    --duration $((PROFILE_DURATION + 15)) --batch-size $EMBED_BATCH_SIZE --seq-length $EMBED_SEQ_LENGTH \
    > "$RESULTS_DIR/p3_embed_output.txt" 2>&1 &
EMBED_PID=$!
sleep 5

python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration $((PROFILE_DURATION + 10)) \
    --concurrency 8 --max-tokens 256 \
    > "$RESULTS_DIR/p3_gen_output.txt" 2>&1 &
GEN_CLIENT_PID=$!

sleep 3

echo "Capturing GPU metrics (time-sliced, ${PROFILE_DURATION}s)..."
nvidia-smi dmon -i $GPU_ID -s uct -d 1 -c $PROFILE_DURATION > "$RESULTS_DIR/dmon_timesliced.txt" 2>&1 &
DMON_PID=$!

wait "$DMON_PID" 2>/dev/null || true
wait "$GEN_CLIENT_PID" 2>/dev/null || true
wait "$EMBED_PID" 2>/dev/null || true
EMBED_PID=""
echo "Profile 3 done."
echo "Gen throughput:"
grep "Throughput:" "$RESULTS_DIR/p3_gen_output.txt" || true
echo "Embed throughput:"
grep "Throughput:" "$RESULTS_DIR/p3_embed_output.txt" || true
stop_vllm

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "GPU Metrics Comparison"
echo "=========================================="
echo ""
echo "--- Gen only (dmon) ---"
cat "$RESULTS_DIR/dmon_gen_only.txt"
echo ""
echo "--- MPS (dmon) ---"
cat "$RESULTS_DIR/dmon_mps.txt"
echo ""
echo "--- Time-sliced (dmon) ---"
cat "$RESULTS_DIR/dmon_timesliced.txt"
echo ""
echo "Results in: $RESULTS_DIR"
