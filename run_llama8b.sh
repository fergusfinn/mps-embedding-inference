#!/bin/bash
set -euo pipefail

# Llama 3.1 8B experiment — more bandwidth-bound model
# Lower batch concurrency to maximize bandwidth-bound decode regime

GPU_ID=1
VLLM_PORT=8100
MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
DURATION=30  # shorter runs since we already have the framework working
GEN_CONCURRENCY=1  # single stream to maximize bandwidth-bound behavior
GEN_MAX_TOKENS=256
EMBED_BATCH_SIZE=32
EMBED_SEQ_LENGTH=128
# Lower GPU memory util to leave room for embedding model (~1.3GB)
GPU_MEM_UTIL=0.85

DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$DIR/results/llama8b_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== Llama 8B Experiment ==="
echo "GPU: $GPU_ID"
echo "Model: $MODEL"
echo "Gen concurrency: $GEN_CONCURRENCY (low to stay bandwidth-bound)"
echo "GPU mem util: $GPU_MEM_UTIL"
echo "Results: $RESULTS_DIR"
echo ""

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
    echo "ERROR: vLLM failed to start after 180s"
    return 1
}

stop_vllm() {
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 2
}

record_gpu_mem() {
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits -i "$GPU_ID" \
        | tee -a "$RESULTS_DIR/gpu_mem_${1}.txt"
}

run_dmon() {
    nvidia-smi dmon -i $GPU_ID -s uct -d 1 -c $DURATION > "$RESULTS_DIR/dmon_${1}.txt" 2>&1 &
    echo $!
}

check_nsys_profile() {
    # Verify that an nsys-rep file was produced
    local profile="$1"
    if [ -f "${profile}.nsys-rep" ]; then
        echo "nsys profile saved: ${profile}.nsys-rep"
    else
        echo "WARNING: ${profile}.nsys-rep not found"
    fi
}

# ============================================================
# STEP 1: Generative-only baseline
# ============================================================
echo "=========================================="
echo "STEP 1: Generative-only baseline"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_ID nsys profile \
    --trace=cuda --force-overwrite=true \
    --output="$RESULTS_DIR/step1_gen_profile" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len 2048 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_step1.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

echo "GPU memory after load:"
record_gpu_mem "step1"

# Warm up
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" --duration 5 --concurrency $GEN_CONCURRENCY --max-tokens 64 --model "$MODEL" > /dev/null 2>&1

DMON_PID=$(run_dmon "step1")

python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    --model "$MODEL" \
    2>&1 | tee "$RESULTS_DIR/step1_gen.txt"

wait "$DMON_PID" 2>/dev/null || true
stop_vllm
check_nsys_profile "$RESULTS_DIR/step1_gen_profile"
echo "Step 1 complete."
echo ""

# ============================================================
# STEP 2: Co-located with MPS
# ============================================================
echo "=========================================="
echo "STEP 2: Co-located with MPS"
echo "=========================================="

MPS_PIPE=/tmp/nvidia-mps-experiment-$GPU_ID
MPS_LOG=/tmp/nvidia-mps-experiment-log-$GPU_ID
mkdir -p "$MPS_PIPE" "$MPS_LOG"
CUDA_VISIBLE_DEVICES=$GPU_ID CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" nvidia-cuda-mps-control -d
sleep 2
echo "MPS started."

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE

nsys profile \
    --trace=cuda --force-overwrite=true \
    --output="$RESULTS_DIR/step2_gen_profile" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len 2048 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_step2.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

echo "GPU memory after load (MPS):"
record_gpu_mem "step2_vllm"

# Warm up gen
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" --duration 5 --concurrency $GEN_CONCURRENCY --max-tokens 64 --model "$MODEL" > /dev/null 2>&1

# Start embedding (also wrapped with nsys)
nsys profile \
    --trace=cuda --force-overwrite=true \
    --output="$RESULTS_DIR/step2_embed_profile" \
    python3 "$DIR/embed_benchmark.py" --duration $((DURATION + 15)) --batch-size $EMBED_BATCH_SIZE --seq-length $EMBED_SEQ_LENGTH \
    > "$RESULTS_DIR/step2_embed.txt" 2>&1 &
EMBED_PID=$!
sleep 5  # let embedding warm up

echo "GPU memory with both (MPS):"
record_gpu_mem "step2_both"

DMON_PID=$(run_dmon "step2")

python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    --model "$MODEL" \
    2>&1 | tee "$RESULTS_DIR/step2_gen.txt"

wait "$DMON_PID" 2>/dev/null || true
wait "$EMBED_PID" 2>/dev/null || true
EMBED_PID=""
echo "Embedding results:"
cat "$RESULTS_DIR/step2_embed.txt"
stop_vllm
check_nsys_profile "$RESULTS_DIR/step2_gen_profile"
check_nsys_profile "$RESULTS_DIR/step2_embed_profile"

echo quit | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control 2>/dev/null || true
unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_VISIBLE_DEVICES
sleep 2
echo "Step 2 complete."
echo ""

# ============================================================
# STEP 3: Co-located without MPS (time-sliced)
# ============================================================
echo "=========================================="
echo "STEP 3: Co-located without MPS (time-sliced)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_ID nsys profile \
    --trace=cuda --force-overwrite=true \
    --output="$RESULTS_DIR/step3_gen_profile" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len 2048 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_step3.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

echo "GPU memory after load:"
record_gpu_mem "step3_vllm"

python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" --duration 5 --concurrency $GEN_CONCURRENCY --max-tokens 64 --model "$MODEL" > /dev/null 2>&1

CUDA_VISIBLE_DEVICES=$GPU_ID nsys profile \
    --trace=cuda --force-overwrite=true \
    --output="$RESULTS_DIR/step3_embed_profile" \
    python3 "$DIR/embed_benchmark.py" \
    --duration $((DURATION + 15)) --batch-size $EMBED_BATCH_SIZE --seq-length $EMBED_SEQ_LENGTH \
    > "$RESULTS_DIR/step3_embed.txt" 2>&1 &
EMBED_PID=$!
sleep 5

echo "GPU memory with both:"
record_gpu_mem "step3_both"

DMON_PID=$(run_dmon "step3")

python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    --model "$MODEL" \
    2>&1 | tee "$RESULTS_DIR/step3_gen.txt"

wait "$DMON_PID" 2>/dev/null || true
wait "$EMBED_PID" 2>/dev/null || true
EMBED_PID=""
echo "Embedding results:"
cat "$RESULTS_DIR/step3_embed.txt"
stop_vllm
check_nsys_profile "$RESULTS_DIR/step3_gen_profile"
check_nsys_profile "$RESULTS_DIR/step3_embed_profile"
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
echo "=== GPU Metrics (dmon) ==="
echo ""
for step in step1 step2 step3; do
    echo "--- $step ---"
    cat "$RESULTS_DIR/dmon_$step.txt"
    echo ""
done

echo ""
echo "=== Kernel Overlap Analysis (nsys) ==="
echo ""
python3 "$DIR/analyze_nsys.py" "$RESULTS_DIR" 2>&1 || echo "nsys analysis failed (check analyze_nsys.py)"

echo ""
echo "Full results in: $RESULTS_DIR"
