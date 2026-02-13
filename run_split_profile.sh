#!/bin/bash
set -euo pipefail

# Profile generative and embedding models independently with DCGM,
# each configured to use ~half the GPU, to see if their resource
# profiles are complementary for MPS co-scheduling.
#
# Step 1: Gen-only (vLLM with gpu_mem_util=0.50, CUDA graphs)
# Step 2: Embed-only (Qwen3-Embedding-8B BF16, direct transformers)

GPU_ID=0
VLLM_PORT=8100
GEN_MODEL="${1:-Qwen/Qwen3-30B-A3B}"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
GEN_MEM_UTIL=0.50          # ~70 GB for vLLM (30 GB weights + 40 GB KV cache)
GEN_CONCURRENCY="${2:-256}"  # realistic serving point
GEN_MAX_TOKENS=512
EMBED_BATCH_SIZE=8
EMBED_SEQ_LENGTH=128
MEASURE_DURATION="${3:-60}"

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv/bin/activate"

GEN_SHORT=$(echo "$GEN_MODEL" | sed 's|.*/||')
RESULTS_DIR="$DIR/results/split_${GEN_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

source "$VENV"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PATH=/usr/local/cuda-12.8/bin:$PATH

echo "=== Split Profile: Gen vs Embed ==="
echo "Gen model: $GEN_MODEL (gpu_mem_util=$GEN_MEM_UTIL)"
echo "Embed model: $EMBED_MODEL (BF16, batch=$EMBED_BATCH_SIZE, seq=$EMBED_SEQ_LENGTH)"
echo "Measure duration: ${MEASURE_DURATION}s"
echo "Results: $RESULTS_DIR"
echo ""

VLLM_PID=""
EMBED_PID=""
DMON_PID=""
DCGM_PID=""

cleanup() {
    [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null || true
    [ -n "${DCGM_PID:-}" ] && kill "$DCGM_PID" 2>/dev/null || true
    [ -n "${EMBED_PID:-}" ] && kill "$EMBED_PID" 2>/dev/null || true
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_vllm() {
    for i in $(seq 1 300); do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "vLLM ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: vLLM failed to start"
    return 1
}

start_monitors() {
    local tag="$1"
    nvidia-smi dmon -i $GPU_ID -s ucp -d 1 -c $((MEASURE_DURATION + 5)) \
        > "$RESULTS_DIR/dmon_${tag}.txt" 2>&1 &
    DMON_PID=$!

    dcgmi dmon -i 0 -e 1002,1003,1004,1005,1007,1008,1013 \
        -d 1000 -c $((MEASURE_DURATION + 5)) \
        > "$RESULTS_DIR/dcgm_${tag}.txt" 2>&1 &
    DCGM_PID=$!
}

stop_monitors() {
    kill "$DMON_PID" 2>/dev/null || true
    wait "$DMON_PID" 2>/dev/null || true
    DMON_PID=""
    kill "$DCGM_PID" 2>/dev/null || true
    wait "$DCGM_PID" 2>/dev/null || true
    DCGM_PID=""
}

print_dcgm_summary() {
    local tag="$1"
    local dcgm_file="$RESULTS_DIR/dcgm_${tag}.txt"
    local dmon_file="$RESULTS_DIR/dmon_${tag}.txt"

    if [ -f "$dcgm_file" ]; then
        local DATA=$(grep "^GPU 0" "$dcgm_file" | tail -10)
        if [ -n "$DATA" ]; then
            echo "  DCGM averages (last 10 samples):"
            local SM_ACT=$(echo "$DATA" | awk '{sum+=$3; n++} END {printf "%.1f%%", sum/n*100}')
            local SM_OCC=$(echo "$DATA" | awk '{sum+=$4; n++} END {printf "%.1f%%", sum/n*100}')
            local TENSOR=$(echo "$DATA" | awk '{sum+=$5; n++} END {printf "%.1f%%", sum/n*100}')
            local DRAM=$(echo "$DATA"   | awk '{sum+=$6; n++} END {printf "%.1f%%", sum/n*100}')
            local FP32=$(echo "$DATA"   | awk '{sum+=$7; n++} END {printf "%.2f%%", sum/n*100}')
            local FP16=$(echo "$DATA"   | awk '{sum+=$8; n++} END {printf "%.2f%%", sum/n*100}')
            echo "    SM active: $SM_ACT  |  SM occ: $SM_OCC  |  Tensor: $TENSOR  |  DRAM BW: $DRAM"
            echo "    FP32: $FP32  |  FP16/BF16: $FP16"
        fi
    fi

    if [ -f "$dmon_file" ]; then
        local DATA=$(grep -E "^\s+0" "$dmon_file" | tail -10)
        if [ -n "$DATA" ]; then
            local PWR=$(echo "$DATA" | awk '{sum+=$10; n++} END {printf "%.0fW", sum/n}')
            local TEMP=$(echo "$DATA" | awk '{sum+=$11; n++} END {printf "%.0fC", sum/n}')
            echo "    Power: $PWR  |  Temp: $TEMP"
        fi
    fi
}

# =========================================
# STEP 1: Gen-only (vLLM, CUDA graphs, half GPU)
# =========================================
echo "=========================================="
echo "STEP 1: Gen-only (C=$GEN_CONCURRENCY, mem_util=$GEN_MEM_UTIL)"
echo "=========================================="

python3 -m vllm.entrypoints.openai.api_server \
    --model "$GEN_MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GEN_MEM_UTIL --max-model-len 4096 \
    --max-num-seqs "$GEN_CONCURRENCY" \
    --quantization fp8 \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_gen.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

# Report VRAM after vLLM startup
echo "VRAM after vLLM startup:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -i $GPU_ID

# Warmup — use full-length outputs to fill KV cache before measurement
echo "Warming up (filling KV cache)..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration 30 --concurrency "$GEN_CONCURRENCY" --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" > /dev/null 2>&1

start_monitors "gen_only"

echo "Measuring gen-only for ${MEASURE_DURATION}s (KV cache should be saturated)..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration "$MEASURE_DURATION" --concurrency "$GEN_CONCURRENCY" --max-tokens "$GEN_MAX_TOKENS" \
    --model "$GEN_MODEL" > "$RESULTS_DIR/bench_gen.txt" 2>&1

stop_monitors

echo ""
echo "--- Gen-only results ---"
grep -E "(Throughput|tok/s)" "$RESULTS_DIR/bench_gen.txt" || tail -5 "$RESULTS_DIR/bench_gen.txt"
print_dcgm_summary "gen_only"
echo ""

# Get KV cache info
grep -E "GPU KV cache usage" "$RESULTS_DIR/vllm_gen.log" | tail -3

# Stop vLLM
kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
VLLM_PID=""
sleep 2

# =========================================
# STEP 2: Embed-only (direct transformers, BF16)
# =========================================
echo ""
echo "=========================================="
echo "STEP 2: Embed-only (batch=$EMBED_BATCH_SIZE, seq=$EMBED_SEQ_LENGTH)"
echo "=========================================="

# Warmup (first few iterations include model loading)
echo "Loading and warming up embedding model..."
python3 "$DIR/embed_benchmark.py" --duration 10 --batch-size "$EMBED_BATCH_SIZE" \
    --seq-length "$EMBED_SEQ_LENGTH" --model "$EMBED_MODEL" \
    > "$RESULTS_DIR/embed_warmup.txt" 2>&1

echo "VRAM after embed warmup:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -i $GPU_ID

start_monitors "embed_only"

echo "Measuring embed-only for ${MEASURE_DURATION}s..."
python3 "$DIR/embed_benchmark.py" --duration "$MEASURE_DURATION" --batch-size "$EMBED_BATCH_SIZE" \
    --seq-length "$EMBED_SEQ_LENGTH" --model "$EMBED_MODEL" \
    > "$RESULTS_DIR/bench_embed.txt" 2>&1

stop_monitors

echo ""
echo "--- Embed-only results ---"
cat "$RESULTS_DIR/bench_embed.txt"
echo ""
print_dcgm_summary "embed_only"

# =========================================
# Summary
# =========================================
echo ""
echo "=========================================="
echo "SUMMARY: Resource profiles for co-scheduling"
echo "=========================================="
echo ""
echo "Gen model:   $GEN_MODEL (FP8, gpu_mem_util=$GEN_MEM_UTIL, C=$GEN_CONCURRENCY)"
echo "Embed model: $EMBED_MODEL (BF16, batch=$EMBED_BATCH_SIZE, seq=$EMBED_SEQ_LENGTH)"
echo ""

printf "%-15s %10s %10s %10s %10s %10s %8s\n" \
    "Workload" "SM act" "SM occ" "Tensor" "DRAM BW" "FP16/BF16" "Power"

for TAG in gen_only embed_only; do
    DCGM="$RESULTS_DIR/dcgm_${TAG}.txt"
    DMON="$RESULTS_DIR/dmon_${TAG}.txt"
    SM_ACT="-" ; SM_OCC="-" ; TENSOR="-" ; DRAM="-" ; FP16="-" ; PWR="-"

    if [ -f "$DCGM" ]; then
        DATA=$(grep "^GPU 0" "$DCGM" | tail -10)
        if [ -n "$DATA" ]; then
            SM_ACT=$(echo "$DATA" | awk '{sum+=$3; n++} END {printf "%.1f%%", sum/n*100}')
            SM_OCC=$(echo "$DATA" | awk '{sum+=$4; n++} END {printf "%.1f%%", sum/n*100}')
            TENSOR=$(echo "$DATA" | awk '{sum+=$5; n++} END {printf "%.1f%%", sum/n*100}')
            DRAM=$(echo "$DATA"   | awk '{sum+=$6; n++} END {printf "%.1f%%", sum/n*100}')
            FP16=$(echo "$DATA"   | awk '{sum+=$8; n++} END {printf "%.2f%%", sum/n*100}')
        fi
    fi
    if [ -f "$DMON" ]; then
        DATA=$(grep -E "^\s+0" "$DMON" | tail -10)
        [ -n "$DATA" ] && PWR=$(echo "$DATA" | awk '{sum+=$10; n++} END {printf "%.0fW", sum/n}')
    fi

    LABEL="gen_only"
    [ "$TAG" = "embed_only" ] && LABEL="embed_only"
    printf "%-15s %10s %10s %10s %10s %10s %8s\n" \
        "$LABEL" "$SM_ACT" "$SM_OCC" "$TENSOR" "$DRAM" "$FP16" "$PWR"
done

echo ""
echo "If SM active + SM active < 100% and DRAM BW + DRAM BW < 100%"
echo "and Power + Power < 700W, co-scheduling should be feasible."
echo ""
echo "Full results in: $RESULTS_DIR"
