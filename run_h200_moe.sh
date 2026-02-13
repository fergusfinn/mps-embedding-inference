#!/bin/bash
set -euo pipefail

# H200 MoE co-scheduling experiment
# Tests whether MoE decode (bandwidth-bound) leaves idle SMs for embedding work

GPU_ID=0
VLLM_PORT=8100
MODEL="Qwen/Qwen3-30B-A3B"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
DURATION=60
GEN_CONCURRENCY=1       # low to maximize bandwidth-bound behavior
GEN_MAX_TOKENS=512      # longer decode to spend more time decoding vs prefilling
EMBED_BATCH_SIZE=8      # conservative for 8B model
EMBED_SEQ_LENGTH=128
GPU_MEM_UTIL=0.45       # leave room for ~16GB embedding model

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv/bin/activate"
RESULTS_DIR="$DIR/results/h200_moe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Activate venv
source "$VENV"

echo "=== H200 MoE Co-Scheduling Experiment ==="
echo "GPU: $GPU_ID (H200 141GB)"
echo "Gen model: $MODEL"
echo "Embed model: $EMBED_MODEL"
echo "Gen concurrency: $GEN_CONCURRENCY"
echo "Gen max tokens: $GEN_MAX_TOKENS"
echo "Embed batch size: $EMBED_BATCH_SIZE"
echo "GPU mem util: $GPU_MEM_UTIL"
echo "Duration: ${DURATION}s per step"
echo "Results: $RESULTS_DIR"
echo ""

VLLM_PID=""
EMBED_PID=""
DMON_PID=""

cleanup() {
    echo "Cleaning up..."
    [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null || true
    [ -n "${EMBED_PID:-}" ] && kill "$EMBED_PID" 2>/dev/null && wait "$EMBED_PID" 2>/dev/null || true
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
    echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-h200 nvidia-cuda-mps-control 2>/dev/null || true
}
trap cleanup EXIT

wait_for_vllm() {
    echo "Waiting for vLLM to start..."
    for i in $(seq 1 300); do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "vLLM ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: vLLM failed to start after 300s"
    echo "Last 50 lines of vLLM log:"
    tail -50 "$RESULTS_DIR/vllm_${1:-unknown}.log" 2>/dev/null || true
    return 1
}

stop_vllm() {
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 3
}

record_gpu_mem() {
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits -i "$GPU_ID" \
        | tee -a "$RESULTS_DIR/gpu_mem_${1}.txt"
}

run_dmon() {
    nvidia-smi dmon -i $GPU_ID -s ucp -d 1 -c $((DURATION + 10)) > "$RESULTS_DIR/dmon_${1}.txt" 2>&1 &
    DMON_PID=$!
    echo "$DMON_PID"
}

stop_dmon() {
    [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null || true
    wait "$DMON_PID" 2>/dev/null || true
    DMON_PID=""
}

# ============================================================
# STEP 1: Generative-only baseline
# ============================================================
echo "=========================================="
echo "STEP 1: Generative-only baseline"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len 4096 \
    --quantization fp8 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_step1.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm "step1"

echo "GPU memory after gen model load:"
record_gpu_mem "step1"

# Warm up
echo "Warming up gen model..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration 10 --concurrency $GEN_CONCURRENCY --max-tokens 64 --model "$MODEL" > /dev/null 2>&1

run_dmon "step1"

python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    --model "$MODEL" \
    2>&1 | tee "$RESULTS_DIR/step1_gen.txt"

stop_dmon
stop_vllm
echo "Step 1 complete."
echo ""

# ============================================================
# STEP 2: Co-located with MPS
# ============================================================
echo "=========================================="
echo "STEP 2: Co-located with MPS"
echo "=========================================="

MPS_PIPE=/tmp/nvidia-mps-h200
MPS_LOG=/tmp/nvidia-mps-h200-log
mkdir -p "$MPS_PIPE" "$MPS_LOG"
CUDA_VISIBLE_DEVICES=$GPU_ID CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" nvidia-cuda-mps-control -d
sleep 2
echo "MPS daemon started."

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len 4096 \
    --quantization fp8 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_step2.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm "step2"

echo "GPU memory after gen model load (MPS):"
record_gpu_mem "step2_vllm"

# Warm up gen
echo "Warming up gen model..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration 10 --concurrency $GEN_CONCURRENCY --max-tokens 64 --model "$MODEL" > /dev/null 2>&1

# Start embedding benchmark (runs longer to overlap fully with gen)
echo "Starting embedding benchmark..."
python3 "$DIR/embed_benchmark.py" \
    --duration $((DURATION + 30)) --batch-size $EMBED_BATCH_SIZE --seq-length $EMBED_SEQ_LENGTH \
    --model "$EMBED_MODEL" \
    > "$RESULTS_DIR/step2_embed.txt" 2>&1 &
EMBED_PID=$!
sleep 10  # let embedding model load and warm up

echo "GPU memory with both models (MPS):"
record_gpu_mem "step2_both"

run_dmon "step2"

python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    --model "$MODEL" \
    2>&1 | tee "$RESULTS_DIR/step2_gen.txt"

stop_dmon

# Signal embedding to stop and print partial results
kill -TERM "$EMBED_PID" 2>/dev/null || true
sleep 2
wait "$EMBED_PID" 2>/dev/null || true
EMBED_PID=""
echo ""
echo "Embedding results (MPS):"
cat "$RESULTS_DIR/step2_embed.txt"
stop_vllm

# Stop MPS
echo quit | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control 2>/dev/null || true
unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_VISIBLE_DEVICES
sleep 3
echo "Step 2 complete."
echo ""

# ============================================================
# STEP 3: Co-located without MPS (time-sliced)
# ============================================================
echo "=========================================="
echo "STEP 3: Co-located without MPS (time-sliced)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len 4096 \
    --quantization fp8 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_step3.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm "step3"

echo "GPU memory after gen model load:"
record_gpu_mem "step3_vllm"

# Warm up gen
echo "Warming up gen model..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration 10 --concurrency $GEN_CONCURRENCY --max-tokens 64 --model "$MODEL" > /dev/null 2>&1

# Start embedding benchmark
echo "Starting embedding benchmark..."
CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$DIR/embed_benchmark.py" \
    --duration $((DURATION + 30)) --batch-size $EMBED_BATCH_SIZE --seq-length $EMBED_SEQ_LENGTH \
    --model "$EMBED_MODEL" \
    > "$RESULTS_DIR/step3_embed.txt" 2>&1 &
EMBED_PID=$!
sleep 10

echo "GPU memory with both models:"
record_gpu_mem "step3_both"

run_dmon "step3"

python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    --model "$MODEL" \
    2>&1 | tee "$RESULTS_DIR/step3_gen.txt"

stop_dmon

# Signal embedding to stop and print partial results
kill -TERM "$EMBED_PID" 2>/dev/null || true
sleep 2
wait "$EMBED_PID" 2>/dev/null || true
EMBED_PID=""
echo ""
echo "Embedding results (time-sliced):"
cat "$RESULTS_DIR/step3_embed.txt"
stop_vllm
echo "Step 3 complete."
echo ""

# ============================================================
# STEP 4: Embedding-only baseline
# ============================================================
echo "=========================================="
echo "STEP 4: Embedding-only baseline"
echo "=========================================="

run_dmon "step4"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$DIR/embed_benchmark.py" \
    --duration "$DURATION" --batch-size $EMBED_BATCH_SIZE --seq-length $EMBED_SEQ_LENGTH \
    --model "$EMBED_MODEL" \
    2>&1 | tee "$RESULTS_DIR/step4_embed.txt"

stop_dmon
echo "Step 4 complete."
echo ""

# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Step 1 (gen only - baseline):"
grep -E "Throughput:|Per-request|Per-token|Decode rate" "$RESULTS_DIR/step1_gen.txt" || true
echo ""
echo "Step 2 (MPS co-located):"
echo "  Gen:"
grep -E "Throughput:|Per-request|Per-token|Decode rate" "$RESULTS_DIR/step2_gen.txt" || true
echo "  Embed:"
grep "Throughput:" "$RESULTS_DIR/step2_embed.txt" || true
echo ""
echo "Step 3 (time-sliced co-located):"
echo "  Gen:"
grep -E "Throughput:|Per-request|Per-token|Decode rate" "$RESULTS_DIR/step3_gen.txt" || true
echo "  Embed:"
grep "Throughput:" "$RESULTS_DIR/step3_embed.txt" || true
echo ""
echo "Step 4 (embed only - baseline):"
grep "Throughput:" "$RESULTS_DIR/step4_embed.txt" || true
echo ""

# Compute gen throughput delta
S1_TPUT=$(grep "Throughput:" "$RESULTS_DIR/step1_gen.txt" 2>/dev/null | grep -oP '[\d.]+(?= tokens/sec)' || echo "0")
S2_TPUT=$(grep "Throughput:" "$RESULTS_DIR/step2_gen.txt" 2>/dev/null | grep -oP '[\d.]+(?= tokens/sec)' || echo "0")
S3_TPUT=$(grep "Throughput:" "$RESULTS_DIR/step3_gen.txt" 2>/dev/null | grep -oP '[\d.]+(?= tokens/sec)' || echo "0")

if [ "$S1_TPUT" != "0" ] && [ "$S2_TPUT" != "0" ] && [ "$S3_TPUT" != "0" ]; then
    echo "=== Gen Throughput Impact ==="
    echo "  Baseline:     $S1_TPUT tok/s"
    echo "  MPS:          $S2_TPUT tok/s ($(python3 -c "print(f'{($S2_TPUT/$S1_TPUT - 1)*100:+.1f}%')" 2>/dev/null || echo '?'))"
    echo "  Time-sliced:  $S3_TPUT tok/s ($(python3 -c "print(f'{($S3_TPUT/$S1_TPUT - 1)*100:+.1f}%')" 2>/dev/null || echo '?'))"
    echo ""
fi

echo "=== GPU Metrics (dmon) ==="
echo ""
for step in step1 step2 step3 step4; do
    if [ -f "$RESULTS_DIR/dmon_$step.txt" ]; then
        echo "--- $step ---"
        head -20 "$RESULTS_DIR/dmon_$step.txt"
        echo "..."
        echo ""
    fi
done

echo ""
echo "Full results in: $RESULTS_DIR"
