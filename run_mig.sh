#!/bin/bash
set -euo pipefail

# MIG experiment: gen on 4g.71gb partition, embed on 3g.71gb partition
# Hard isolation baseline — no resource sharing between partitions
#
# Prerequisite: MIG enabled with 4g.71gb + 3g.71gb instances created:
#   sudo nvidia-smi -i 0 -mig 1
#   sudo nvidia-smi mig -i 0 -cgi 5,9
#   sudo nvidia-smi mig -i 0 -gi 1 -cci 0
#   sudo nvidia-smi mig -i 0 -gi 2 -cci 0

VLLM_PORT=8100
GEN_MODEL="${1:-Qwen/Qwen3-30B-A3B-FP8}"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
GEN_MEM_UTIL=0.85
GEN_CONCURRENCY=1024
GEN_MAX_TOKENS=512
EMBED_BATCH_SIZE=8
EMBED_SEQ_LENGTH=128
WARMUP_DURATION=30
MEASURE_DURATION=60

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv/bin/activate"

GEN_SHORT=$(echo "$GEN_MODEL" | sed 's|.*/||')
RESULTS_DIR="$DIR/results/mig_${GEN_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

source "$VENV"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Discover MIG device UUIDs
MIG_4G=$(nvidia-smi -L | grep '1c.4g.71gb' | grep -oP 'MIG-[a-f0-9-]+')
MIG_3G=$(nvidia-smi -L | grep '1c.3g.71gb' | grep -oP 'MIG-[a-f0-9-]+')

if [ -z "$MIG_4G" ] || [ -z "$MIG_3G" ]; then
    echo "ERROR: Could not find MIG devices. Run the setup commands first."
    echo "  sudo nvidia-smi -i 0 -mig 1"
    echo "  sudo nvidia-smi mig -i 0 -cgi 5,9"
    echo "  sudo nvidia-smi mig -i 0 -gi 1 -cci 0"
    echo "  sudo nvidia-smi mig -i 0 -gi 2 -cci 0"
    exit 1
fi

echo "=== MIG Isolation Experiment ==="
echo "Gen:   $GEN_MODEL (FP8, mem_util=$GEN_MEM_UTIL, C=$GEN_CONCURRENCY)"
echo "Embed: $EMBED_MODEL (BF16, batch=$EMBED_BATCH_SIZE, seq=$EMBED_SEQ_LENGTH)"
echo "Gen partition:   4g.71gb (UUID: $MIG_4G)"
echo "Embed partition: 3g.71gb (UUID: $MIG_3G)"
echo "Warmup: ${WARMUP_DURATION}s, Measure: ${MEASURE_DURATION}s"
echo "Results: $RESULTS_DIR"
echo ""

VLLM_PID=""
EMBED_PID=""

cleanup() {
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

# =========================================
# STEP 1: Gen-only on 4g.71gb partition
# =========================================
echo "=========================================="
echo "STEP 1: Gen-only on MIG 4g.71gb (64 SMs, 70 GB)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$MIG_4G python3 -m vllm.entrypoints.openai.api_server \
    --model "$GEN_MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GEN_MEM_UTIL --max-model-len 4096 \
    --max-num-seqs "$GEN_CONCURRENCY" \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_step1.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

echo "Warming up gen (${WARMUP_DURATION}s)..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration "$WARMUP_DURATION" --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" > /dev/null 2>&1

echo "Measuring gen-only for ${MEASURE_DURATION}s..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration "$MEASURE_DURATION" --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" \
    > "$RESULTS_DIR/bench_step1_gen.txt" 2>&1

echo "--- Step 1: Gen-only on MIG 4g.71gb ---"
grep -E "Throughput:" "$RESULTS_DIR/bench_step1_gen.txt"
grep -E "p50:|p95:|p99:|Decode rate" "$RESULTS_DIR/bench_step1_gen.txt" || true
echo ""

kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
VLLM_PID=""
sleep 2

# =========================================
# STEP 2: Gen + Embed on separate MIG partitions
# =========================================
echo "=========================================="
echo "STEP 2: Gen (4g.71gb) + Embed (3g.71gb) — MIG isolated"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$MIG_4G python3 -m vllm.entrypoints.openai.api_server \
    --model "$GEN_MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GEN_MEM_UTIL --max-model-len 4096 \
    --max-num-seqs "$GEN_CONCURRENCY" \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm_step2.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

echo "Warming up gen (${WARMUP_DURATION}s)..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration "$WARMUP_DURATION" --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" > /dev/null 2>&1

# Start embedding on the 3g partition
echo "Starting embed benchmark on MIG 3g.71gb..."
CUDA_VISIBLE_DEVICES=$MIG_3G python3 "$DIR/embed_benchmark.py" \
    --duration $((MEASURE_DURATION + 5)) --batch-size "$EMBED_BATCH_SIZE" \
    --seq-length "$EMBED_SEQ_LENGTH" --model "$EMBED_MODEL" \
    > "$RESULTS_DIR/bench_step2_embed.txt" 2>&1 &
EMBED_PID=$!
sleep 5  # let embed load

echo "Measuring gen+embed (MIG isolated) for ${MEASURE_DURATION}s..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration "$MEASURE_DURATION" --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" --model "$GEN_MODEL" \
    > "$RESULTS_DIR/bench_step2_gen.txt" 2>&1

kill "$EMBED_PID" 2>/dev/null || true
wait "$EMBED_PID" 2>/dev/null || true
EMBED_PID=""

echo "--- Step 2: Gen + Embed (MIG isolated) ---"
echo "Gen:"
grep -E "Throughput:" "$RESULTS_DIR/bench_step2_gen.txt"
grep -E "p50:|p95:|p99:|Decode rate" "$RESULTS_DIR/bench_step2_gen.txt" || true
echo "Embed:"
grep -E "Throughput:" "$RESULTS_DIR/bench_step2_embed.txt" || tail -3 "$RESULTS_DIR/bench_step2_embed.txt"
echo ""

kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
VLLM_PID=""

# =========================================
# FINAL SUMMARY
# =========================================
echo ""
echo "=========================================="
echo "MIG EXPERIMENT SUMMARY"
echo "=========================================="
echo ""
echo "Partitions: 4g.71gb (4/7 SMs, 4/8 mem) + 3g.71gb (3/7 SMs, 4/8 mem)"
echo "Note: 4/7 + 3/7 = all SMs allocated. Both partitions get 4/8 = 70 GB VRAM."
echo ""

GEN1=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_step1_gen.txt" 2>/dev/null || echo "0")
GEN2=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_step2_gen.txt" 2>/dev/null || echo "0")
EMB2=$(grep -oP 'Throughput: \K[0-9.]+' "$RESULTS_DIR/bench_step2_embed.txt" 2>/dev/null || echo "0")

if [ "$GEN1" != "0" ] && [ "$GEN2" != "0" ]; then
    DELTA=$(python3 -c "print(f'{($GEN2/$GEN1 - 1)*100:+.1f}%')")
else
    DELTA="-"
fi

printf "%-40s %15s %15s %10s\n" "" "Gen (tok/s)" "Embed (seq/s)" "Gen delta"
printf "%-40s %15s %15s %10s\n" "Step 1: Gen only (4g.71gb)" "$GEN1" "-" "baseline"
printf "%-40s %15s %15s %10s\n" "Step 2: Gen+Embed (MIG isolated)" "$GEN2" "$EMB2" "$DELTA"

echo ""
echo "Per-request latency:"
for STEP_FILE in step1 step2; do
    BENCH="$RESULTS_DIR/bench_${STEP_FILE}_gen.txt"
    if [ -f "$BENCH" ]; then
        echo "  ${STEP_FILE}: $(grep 'Per-request latency' -A1 "$BENCH" 2>/dev/null | tail -1 || echo 'no latency data')"
    fi
done

echo ""
echo "Full results in: $RESULTS_DIR"
