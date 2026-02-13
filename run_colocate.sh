#!/bin/bash
set -euo pipefail

# Co-location experiment: gen + embed on same GPU
# Step 1:  Gen-only baseline (no MPS)
# Step 1b: Gen-only WITH MPS (isolate MPS daemon overhead)
# Step 2:  Co-located WITH MPS (parallel kernel execution)
# Step 3:  Co-located WITHOUT MPS (time-sliced)

GPU_ID=0
VLLM_PORT=8100
GEN_MODEL="${1:-Qwen/Qwen3-30B-A3B}"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
GEN_MEM_UTIL=0.50
GEN_CONCURRENCY=1024
GEN_MAX_TOKENS=512
EMBED_BATCH_SIZE=8
EMBED_SEQ_LENGTH=128
WARMUP_DURATION=30
MEASURE_DURATION=60

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv/bin/activate"

GEN_SHORT=$(echo "$GEN_MODEL" | sed 's|.*/||')
RESULTS_DIR="$DIR/results/colocate_${GEN_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

source "$VENV"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PATH=/usr/local/cuda-12.8/bin:$PATH

echo "=== Co-location Experiment ==="
echo "Gen:   $GEN_MODEL (FP8, mem_util=$GEN_MEM_UTIL, C=$GEN_CONCURRENCY)"
echo "Embed: $EMBED_MODEL (BF16, batch=$EMBED_BATCH_SIZE, seq=$EMBED_SEQ_LENGTH)"
echo "Warmup: ${WARMUP_DURATION}s, Measure: ${MEASURE_DURATION}s"
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
    # Clean up MPS if running
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
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
    nvidia-smi dmon -i $GPU_ID -s ucp -d 1 -c $((MEASURE_DURATION + 10)) \
        > "$RESULTS_DIR/dmon_${tag}.txt" 2>&1 &
    DMON_PID=$!
    dcgmi dmon -i 0 -e 1002,1003,1004,1005,1007,1008,1013 \
        -d 1000 -c $((MEASURE_DURATION + 10)) \
        > "$RESULTS_DIR/dcgm_${tag}.txt" 2>&1 &
    DCGM_PID=$!
}

stop_monitors() {
    kill "$DMON_PID" 2>/dev/null || true; wait "$DMON_PID" 2>/dev/null || true; DMON_PID=""
    kill "$DCGM_PID" 2>/dev/null || true; wait "$DCGM_PID" 2>/dev/null || true; DCGM_PID=""
}

print_summary() {
    local tag="$1"
    local dcgm="$RESULTS_DIR/dcgm_${tag}.txt"
    local dmon="$RESULTS_DIR/dmon_${tag}.txt"

    if [ -f "$dcgm" ]; then
        # Filter to steady-state: SM between 20% and 95% to exclude startup/drain
        local DATA=$(grep "^GPU 0" "$dcgm" | awk '$3 > 0.20 && $3 < 0.95')
        local N=$(echo "$DATA" | wc -l)
        if [ "$N" -gt 5 ]; then
            echo "  DCGM ($N steady-state samples):"
            echo "    SM active: $(echo "$DATA" | awk '{s+=$3;n++} END{printf "%.1f%%",s/n*100}')"
            echo "    SM occ:    $(echo "$DATA" | awk '{s+=$4;n++} END{printf "%.1f%%",s/n*100}')"
            echo "    Tensor:    $(echo "$DATA" | awk '{s+=$5;n++} END{printf "%.1f%%",s/n*100}')"
            echo "    DRAM BW:   $(echo "$DATA" | awk '{s+=$6;n++} END{printf "%.1f%%",s/n*100}')"
        else
            echo "  DCGM (all samples, $N):"
            DATA=$(grep "^GPU 0" "$dcgm" | tail -10)
            echo "    SM active: $(echo "$DATA" | awk '{s+=$3;n++} END{printf "%.1f%%",s/n*100}')"
            echo "    SM occ:    $(echo "$DATA" | awk '{s+=$4;n++} END{printf "%.1f%%",s/n*100}')"
            echo "    Tensor:    $(echo "$DATA" | awk '{s+=$5;n++} END{printf "%.1f%%",s/n*100}')"
            echo "    DRAM BW:   $(echo "$DATA" | awk '{s+=$6;n++} END{printf "%.1f%%",s/n*100}')"
        fi
    fi
    if [ -f "$dmon" ]; then
        local DATA=$(grep -E "^\s+0" "$dmon" | tail -20)
        if [ -n "$DATA" ]; then
            echo "    Power:     $(echo "$DATA" | awk '{s+=$10;n++} END{printf "%.0fW",s/n}')"
            echo "    Temp:      $(echo "$DATA" | awk '{s+=$11;n++} END{printf "%.0fC",s/n}')"
        fi
    fi
}

start_vllm() {
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$GEN_MODEL" --port "$VLLM_PORT" \
        --gpu-memory-utilization $GEN_MEM_UTIL --max-model-len 4096 \
        --max-num-seqs "$GEN_CONCURRENCY" \
        --quantization fp8 \
        --disable-log-requests \
        > "$RESULTS_DIR/vllm_${1}.log" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm
}

stop_vllm() {
    kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 2
}

warmup_gen() {
    echo "Warming up gen (filling KV cache, ${WARMUP_DURATION}s)..."
    python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
        --duration "$WARMUP_DURATION" --concurrency "$GEN_CONCURRENCY" \
        --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" > /dev/null 2>&1
}

# =========================================
# STEP 1: Gen-only baseline
# =========================================
echo "=========================================="
echo "STEP 1: Gen-only baseline"
echo "=========================================="

start_vllm "step1"
warmup_gen
start_monitors "step1_gen_only"

echo "Measuring gen-only for ${MEASURE_DURATION}s..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration "$MEASURE_DURATION" --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" \
    > "$RESULTS_DIR/bench_step1_gen.txt" 2>&1

stop_monitors

echo "--- Step 1 results ---"
grep -E "Throughput:" "$RESULTS_DIR/bench_step1_gen.txt"
grep -E "p50:|p95:|p99:|Decode rate" "$RESULTS_DIR/bench_step1_gen.txt" || true
print_summary "step1_gen_only"
echo ""

stop_vllm

# =========================================
# STEP 1b: Gen-only WITH MPS (overhead control)
# =========================================
echo "=========================================="
echo "STEP 1b: Gen-only WITH MPS (overhead control)"
echo "=========================================="

echo "Starting MPS daemon..."
nvidia-cuda-mps-control -d
sleep 2
echo "MPS active: $(nvidia-cuda-mps-control -s 2>&1 || echo 'unknown')"

start_vllm "step1b"
warmup_gen
start_monitors "step1b_mps_only"

echo "Measuring gen-only (MPS active, no embed) for ${MEASURE_DURATION}s..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration "$MEASURE_DURATION" --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" \
    > "$RESULTS_DIR/bench_step1b_gen.txt" 2>&1

stop_monitors

echo "--- Step 1b results (MPS, no embed) ---"
grep -E "Throughput:" "$RESULTS_DIR/bench_step1b_gen.txt"
grep -E "p50:|p95:|p99:|Decode rate" "$RESULTS_DIR/bench_step1b_gen.txt" || true
print_summary "step1b_mps_only"
echo ""

stop_vllm
# Keep MPS running for Step 2

# =========================================
# STEP 2: Co-located WITH MPS
# =========================================
echo "=========================================="
echo "STEP 2: Co-located WITH MPS"
echo "=========================================="

# MPS already running from step 1b
echo "MPS still active: $(nvidia-cuda-mps-control -s 2>&1 || echo 'unknown')"

start_vllm "step2"
warmup_gen

# Start embedding model in background
echo "Starting embed benchmark in background..."
python3 "$DIR/embed_benchmark.py" \
    --duration $((MEASURE_DURATION + 5)) --batch-size "$EMBED_BATCH_SIZE" \
    --seq-length "$EMBED_SEQ_LENGTH" --model "$EMBED_MODEL" \
    > "$RESULTS_DIR/bench_step2_embed.txt" 2>&1 &
EMBED_PID=$!
sleep 5  # let embed model load and start inference

start_monitors "step2_mps"

echo "Measuring gen+embed (MPS) for ${MEASURE_DURATION}s..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration "$MEASURE_DURATION" --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" \
    > "$RESULTS_DIR/bench_step2_gen.txt" 2>&1

stop_monitors

# Wait for embed to finish
kill "$EMBED_PID" 2>/dev/null || true
wait "$EMBED_PID" 2>/dev/null || true
EMBED_PID=""

echo "--- Step 2 results (MPS) ---"
echo "Gen:"
grep -E "Throughput:" "$RESULTS_DIR/bench_step2_gen.txt"
grep -E "p50:|p95:|p99:|Decode rate" "$RESULTS_DIR/bench_step2_gen.txt" || true
echo "Embed:"
grep -E "Throughput:" "$RESULTS_DIR/bench_step2_embed.txt" || tail -3 "$RESULTS_DIR/bench_step2_embed.txt"
print_summary "step2_mps"
echo ""

stop_vllm

# Stop MPS
echo "Stopping MPS daemon..."
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 2

# =========================================
# STEP 3: Co-located WITHOUT MPS (time-sliced)
# =========================================
echo "=========================================="
echo "STEP 3: Co-located WITHOUT MPS (time-sliced)"
echo "=========================================="

start_vllm "step3"
warmup_gen

# Start embedding model in background (no MPS this time)
echo "Starting embed benchmark in background (no MPS)..."
python3 "$DIR/embed_benchmark.py" \
    --duration $((MEASURE_DURATION + 5)) --batch-size "$EMBED_BATCH_SIZE" \
    --seq-length "$EMBED_SEQ_LENGTH" --model "$EMBED_MODEL" \
    > "$RESULTS_DIR/bench_step3_embed.txt" 2>&1 &
EMBED_PID=$!
sleep 5

start_monitors "step3_no_mps"

echo "Measuring gen+embed (no MPS) for ${MEASURE_DURATION}s..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration "$MEASURE_DURATION" --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" \
    > "$RESULTS_DIR/bench_step3_gen.txt" 2>&1

stop_monitors

kill "$EMBED_PID" 2>/dev/null || true
wait "$EMBED_PID" 2>/dev/null || true
EMBED_PID=""

echo "--- Step 3 results (no MPS) ---"
echo "Gen:"
grep -E "Throughput:" "$RESULTS_DIR/bench_step3_gen.txt"
grep -E "p50:|p95:|p99:|Decode rate" "$RESULTS_DIR/bench_step3_gen.txt" || true
echo "Embed:"
grep -E "Throughput:" "$RESULTS_DIR/bench_step3_embed.txt" || tail -3 "$RESULTS_DIR/bench_step3_embed.txt"
print_summary "step3_no_mps"
echo ""

stop_vllm

# =========================================
# FINAL SUMMARY
# =========================================
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo ""

# Extract throughputs
GEN1=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_step1_gen.txt" 2>/dev/null || echo "0")
GEN1B=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_step1b_gen.txt" 2>/dev/null || echo "0")
GEN2=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_step2_gen.txt" 2>/dev/null || echo "0")
GEN3=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_step3_gen.txt" 2>/dev/null || echo "0")
EMB2=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_step2_embed.txt" 2>/dev/null || echo "0")
EMB3=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_step3_embed.txt" 2>/dev/null || echo "0")

printf "%-30s %15s %15s %10s\n" "" "Gen (tok/s)" "Embed (seq/s)" "Gen delta"
printf "%-30s %15s %15s %10s\n" "Step 1:  Gen only" "$GEN1" "-" "baseline"

if [ "$GEN1" != "0" ] && [ "$GEN1B" != "0" ]; then
    DELTA1B=$(python3 -c "print(f'{($GEN1B/$GEN1 - 1)*100:+.1f}%')")
else
    DELTA1B="-"
fi
printf "%-30s %15s %15s %10s\n" "Step 1b: Gen only (MPS on)" "$GEN1B" "-" "$DELTA1B"

if [ "$GEN1" != "0" ] && [ "$GEN2" != "0" ]; then
    DELTA2=$(python3 -c "print(f'{($GEN2/$GEN1 - 1)*100:+.1f}%')")
else
    DELTA2="-"
fi
printf "%-30s %15s %15s %10s\n" "Step 2:  Gen+Embed (MPS)" "$GEN2" "$EMB2" "$DELTA2"

if [ "$GEN1" != "0" ] && [ "$GEN3" != "0" ]; then
    DELTA3=$(python3 -c "print(f'{($GEN3/$GEN1 - 1)*100:+.1f}%')")
else
    DELTA3="-"
fi
printf "%-30s %15s %15s %10s\n" "Step 3:  Gen+Embed (no MPS)" "$GEN3" "$EMB3" "$DELTA3"

echo ""
echo "Per-request latency:"
for STEP_FILE in step1 step1b step2 step3; do
    BENCH="$RESULTS_DIR/bench_${STEP_FILE}_gen.txt"
    if [ -f "$BENCH" ]; then
        LABEL=$(head -1 "$BENCH" 2>/dev/null || echo "$STEP_FILE")
        LATENCY=$(grep -oP 'p50: \K[0-9.]+s' "$BENCH" 2>/dev/null | head -1 || echo "-")
        P95=$(grep -oP 'p95: \K[0-9.]+s' "$BENCH" 2>/dev/null | head -1 || echo "-")
        P99=$(grep -oP 'p99: \K[0-9.]+s' "$BENCH" 2>/dev/null | head -1 || echo "-")
        echo "  ${STEP_FILE}: $(grep 'Per-request latency' -A1 "$BENCH" 2>/dev/null | tail -1 || echo 'no latency data')"
    fi
done

echo ""
echo "DCGM profiles:"
for STEP in step1_gen_only step1b_mps_only step2_mps step3_no_mps; do
    echo ""
    echo "  $STEP:"
    print_summary "$STEP"
done

echo ""
echo "Full results in: $RESULTS_DIR"
