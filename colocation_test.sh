#!/bin/bash
set -euo pipefail

# Co-location benchmark: both models on full GPU simultaneously
# Goal: beat MIG baselines (gen: 15,007 tok/s, embed: 507 seq/s)
#
# Tests time-sliced sharing (default GPU mode, no MPS).

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"
GEN_MODEL="Qwen/Qwen3-30B-A3B-FP8"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$DIR/colocation_results"
mkdir -p "$RESULTS"

source "$DIR/venv/bin/activate"

DURATION=60

full_cleanup() {
    docker stop vllm-gen vllm-embed 2>/dev/null || true
    docker rm vllm-gen vllm-embed 2>/dev/null || true
    pkill -f gen_benchmark || true
    pkill -f embed_benchmark_vllm || true
}
trap full_cleanup EXIT

wait_health() {
    local port=$1 label=$2
    for i in $(seq 1 300); do
        curl -s "http://localhost:$port/health" > /dev/null 2>&1 && echo "  $label ready after ${i}s" && return
        sleep 1
    done
    echo "  $label FAILED to start" && exit 1
}

run_colocation_test() {
    local label=$1
    local gen_extra=$2
    local embed_extra=$3
    local gen_concurrency=$4
    local embed_concurrency=$5
    local embed_batch=$6
    local gen_gpu_mem=$7
    local embed_gpu_mem=$8

    echo ""
    echo "=== Condition: $label ==="
    echo "  gen: gpu_mem=$gen_gpu_mem, $gen_extra, C=$gen_concurrency"
    echo "  embed: gpu_mem=$embed_gpu_mem, $embed_extra, C=$embed_concurrency, B=$embed_batch"
    full_cleanup
    sleep 2

    # Start gen server
    docker run --rm -d --name vllm-gen \
        --gpus all --network host --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        -e HF_HUB_OFFLINE=1 \
        "$VLLM_IMAGE" "$GEN_MODEL" \
        --port 8100 --gpu-memory-utilization "$gen_gpu_mem" \
        --max-model-len 4096 --disable-log-requests \
        $gen_extra \
        > /dev/null 2>&1

    wait_health 8100 "Gen"

    # Start embed server
    docker run --rm -d --name vllm-embed \
        --gpus all --network host --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        -e HF_HUB_OFFLINE=1 \
        "$VLLM_IMAGE" "$EMBED_MODEL" \
        --port 8200 --gpu-memory-utilization "$embed_gpu_mem" \
        --max-model-len 512 --convert embed \
        --disable-log-requests \
        $embed_extra \
        > /dev/null 2>&1

    wait_health 8200 "Embed"

    # Warmup both
    echo "  warmup 20s (concurrent)..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 20 --concurrency "$gen_concurrency" --max-tokens 512 \
        --model "$GEN_MODEL" > /dev/null 2>&1 &
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 20 --concurrency "$embed_concurrency" --batch-size "$embed_batch" \
        --model "$EMBED_MODEL" > /dev/null 2>&1 &
    wait

    # Measurement: run both concurrently
    echo "  measuring ${DURATION}s (concurrent)..."

    # Capture dmon during measurement
    nvidia-smi dmon -i 0 -s pucvmet -d 1 > "$RESULTS/dmon_${label}.txt" 2>/dev/null &
    DMON_PID=$!

    local embed_result_file="$RESULTS/embed_${label}.txt"
    local gen_result_file="$RESULTS/gen_${label}.txt"

    # Start embed benchmark (runs slightly longer to overlap fully)
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration $((DURATION + 5)) --concurrency "$embed_concurrency" \
        --batch-size "$embed_batch" --model "$EMBED_MODEL" > "$embed_result_file" 2>&1 &
    EMBED_PID=$!

    # Run gen benchmark
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration "$DURATION" --concurrency "$gen_concurrency" --max-tokens 512 \
        --model "$GEN_MODEL" > "$gen_result_file" 2>&1

    wait $EMBED_PID || true
    kill $DMON_PID 2>/dev/null || true

    # Extract results
    local gen_throughput embed_throughput gen_decode
    gen_throughput=$(grep "Throughput:" "$gen_result_file" | awk '{print $2}')
    gen_decode=$(grep "Decode rate" "$gen_result_file" | awk '{print $(NF-1)}')
    embed_throughput=$(grep "Throughput:" "$embed_result_file" | awk '{print $2}')

    echo "  Gen:   $gen_throughput tok/s (decode p50: $gen_decode tok/s/req)"
    echo "  Embed: $embed_throughput seq/s"

    # Compare to MIG baselines
    local gen_pct embed_pct
    gen_pct=$(python3 -c "print(f'{($gen_throughput / 15007 - 1) * 100:+.1f}%')")
    embed_pct=$(python3 -c "print(f'{($embed_throughput / 507 - 1) * 100:+.1f}%')")
    echo "  vs MIG: gen $gen_pct, embed $embed_pct"
}

echo "# Co-location Benchmark"
echo "MIG baselines: gen 15,007 tok/s, embed 507 seq/s"

# Test 1: Standard config (50/30 GPU mem split) — matches prior experiments
run_colocation_test "ts_50_30_c512" \
    "--max-num-seqs 512" "--max-num-seqs 256" \
    512 128 32 0.50 0.30

# Test 2: Give gen more memory (60/20)
run_colocation_test "ts_60_20_c512" \
    "--max-num-seqs 512" "--max-num-seqs 256" \
    512 128 32 0.60 0.20

# Test 3: Even more for gen (65/15)
run_colocation_test "ts_65_15_c512" \
    "--max-num-seqs 512" "--max-num-seqs 256" \
    512 128 32 0.65 0.15

# Test 4: 50/30 with C=256 for gen (was optimal per-token on MIG)
run_colocation_test "ts_50_30_c256" \
    "--max-num-seqs 256" "--max-num-seqs 256" \
    256 128 32 0.50 0.30

# Test 5: Higher embed concurrency
run_colocation_test "ts_50_30_c512_ec256" \
    "--max-num-seqs 512" "--max-num-seqs 256" \
    512 256 32 0.50 0.30

echo ""
echo "=== All co-location tests complete ==="
