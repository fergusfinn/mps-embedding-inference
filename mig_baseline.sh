#!/bin/bash
set -euo pipefail

# MIG baseline benchmark: find maximum throughput for gen and embed models
# on their respective MIG partitions.
#
# MIG layout:
#   Device 0: MIG 4g.71gb (64 SMs, ~71 GB) — gen model (Qwen3-30B-A3B-FP8)
#   Device 1: MIG 3g.71gb (60 SMs, ~71 GB) — embed model (Qwen3-Embedding-8B)
#
# For gen: we optimize output token throughput (tok/s)
# For embed: we optimize sequence throughput (seq/s)

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"
GEN_MODEL="Qwen/Qwen3-30B-A3B-FP8"
EMBED_MODEL="Qwen/Qwen3-Embedding-8B"
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$DIR/mig_baseline_results"
mkdir -p "$RESULTS"

GEN_MIG_UUID="MIG-4f79e892-6606-5a13-81f5-89bec3a53ca6"
EMBED_MIG_UUID="MIG-8de4c802-0943-5681-bf2b-654efd76103d"

source "$DIR/venv/bin/activate"

DURATION=60
NOTES="$RESULTS/notes.md"

log() {
    echo "$1" | tee -a "$NOTES"
}

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

# ============================================================
# PART 1: Gen model on MIG 4g.71gb
# ============================================================
run_gen_test() {
    local label=$1
    local extra_vllm_args=$2
    local concurrency=$3
    local max_tokens=$4

    log ""
    log "### Gen test: $label"
    log "  vLLM args: $extra_vllm_args"
    log "  concurrency=$concurrency, max_tokens=$max_tokens, duration=${DURATION}s"

    full_cleanup
    sleep 2

    docker run --rm -d --name vllm-gen \
        --gpus "\"device=$GEN_MIG_UUID\"" \
        --network host --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        -e HF_HUB_OFFLINE=1 \
        "$VLLM_IMAGE" "$GEN_MODEL" \
        --port 8100 --gpu-memory-utilization 0.90 \
        --max-model-len 4096 --disable-log-requests \
        $extra_vllm_args \
        > /dev/null 2>&1

    wait_health 8100 "Gen"

    # Warmup
    echo "  warmup 15s..."
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 15 --concurrency "$concurrency" --max-tokens "$max_tokens" \
        --model "$GEN_MODEL" > /dev/null 2>&1

    # Measure
    echo "  measuring ${DURATION}s..."
    local result
    result=$(python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration "$DURATION" --concurrency "$concurrency" --max-tokens "$max_tokens" \
        --model "$GEN_MODEL" 2>&1)

    local throughput
    throughput=$(echo "$result" | grep "Throughput:" | awk '{print $2}')
    local decode_rate
    decode_rate=$(echo "$result" | grep "Decode rate" | awk '{print $NF}')

    log "  **Result: $throughput tok/s, decode p50: $decode_rate tok/s/req**"
    echo "$result" >> "$RESULTS/gen_${label}.txt"

    # Capture dmon during a short additional run
    echo "  capturing dmon 20s..."
    nvidia-smi dmon -i 0 -s pucvmet -d 1 > "$RESULTS/dmon_gen_${label}.txt" 2>/dev/null &
    DMON_PID=$!
    python3 "$DIR/gen_benchmark.py" --base-url http://localhost:8100 \
        --duration 20 --concurrency "$concurrency" --max-tokens "$max_tokens" \
        --model "$GEN_MODEL" > /dev/null 2>&1
    kill $DMON_PID 2>/dev/null || true

    docker stop vllm-gen 2>/dev/null || true
    sleep 2
}

# ============================================================
# PART 2: Embed model on MIG 3g.71gb
# ============================================================
run_embed_test() {
    local label=$1
    local extra_vllm_args=$2
    local concurrency=$3
    local batch_size=$4

    log ""
    log "### Embed test: $label"
    log "  vLLM args: $extra_vllm_args"
    log "  concurrency=$concurrency, batch_size=$batch_size, duration=${DURATION}s"

    full_cleanup
    sleep 2

    docker run --rm -d --name vllm-embed \
        --gpus "\"device=$EMBED_MIG_UUID\"" \
        --network host --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$VLLM_CACHE:/root/.cache/vllm" \
        -e HF_HUB_OFFLINE=1 \
        "$VLLM_IMAGE" "$EMBED_MODEL" \
        --port 8200 --gpu-memory-utilization 0.90 \
        --max-model-len 512 --convert embed \
        --disable-log-requests \
        $extra_vllm_args \
        > /dev/null 2>&1

    wait_health 8200 "Embed"

    # Warmup
    echo "  warmup 15s..."
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 15 --concurrency "$concurrency" --batch-size "$batch_size" \
        --model "$EMBED_MODEL" > /dev/null 2>&1

    # Measure
    echo "  measuring ${DURATION}s..."
    local result
    result=$(python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration "$DURATION" --concurrency "$concurrency" --batch-size "$batch_size" \
        --model "$EMBED_MODEL" 2>&1)

    local throughput
    throughput=$(echo "$result" | grep "Throughput:" | awk '{print $2}')

    log "  **Result: $throughput seq/s**"
    echo "$result" >> "$RESULTS/embed_${label}.txt"

    # Capture dmon during a short additional run
    echo "  capturing dmon 20s..."
    nvidia-smi dmon -i 0 -s pucvmet -d 1 > "$RESULTS/dmon_embed_${label}.txt" 2>/dev/null &
    DMON_PID=$!
    python3 "$DIR/embed_benchmark_vllm.py" --base-url http://localhost:8200 \
        --duration 20 --concurrency "$concurrency" --batch-size "$batch_size" \
        --model "$EMBED_MODEL" > /dev/null 2>&1
    kill $DMON_PID 2>/dev/null || true

    docker stop vllm-embed 2>/dev/null || true
    sleep 2
}

# ============================================================
# Run experiments
# ============================================================

echo "# MIG Baseline Benchmark" > "$NOTES"
echo "Date: $(date)" >> "$NOTES"
echo "" >> "$NOTES"
log "## GPU Configuration"
log "MIG 4g.71gb ($GEN_MIG_UUID) — gen model"
log "MIG 3g.71gb ($EMBED_MIG_UUID) — embed model"

log ""
log "## Part 1: Gen Model (Qwen3-30B-A3B-FP8) on MIG 4g.71gb"

# Test 1: Default settings, various concurrencies
run_gen_test "default_c64" "--max-num-seqs 256" 64 512
run_gen_test "default_c256" "--max-num-seqs 256" 256 512
run_gen_test "default_c512" "--max-num-seqs 512" 512 512

# Test 2: Eager mode comparison
run_gen_test "eager_c256" "--max-num-seqs 256 --enforce-eager" 256 512

# Test 3: Higher batch, shorter output
run_gen_test "default_c256_tok256" "--max-num-seqs 256" 256 256

log ""
log "## Part 2: Embed Model (Qwen3-Embedding-8B) on MIG 3g.71gb"

# Test various concurrencies and batch sizes
run_embed_test "c64_b8" "--max-num-seqs 256" 64 8
run_embed_test "c64_b32" "--max-num-seqs 256" 64 32
run_embed_test "c128_b8" "--max-num-seqs 256" 128 8
run_embed_test "c128_b32" "--max-num-seqs 256" 128 32
run_embed_test "c256_b8" "--max-num-seqs 256" 256 8

# Eager mode comparison
run_embed_test "eager_c128_b8" "--max-num-seqs 256 --enforce-eager" 128 8

log ""
log "## Summary"
log "See individual result files in $RESULTS/"

echo ""
echo "=== All tests complete ==="
