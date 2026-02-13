#!/bin/bash
set -euo pipefail

# nsys kernel trace of MoE decode on H200
# Uses VLLM_ENABLE_V1_MULTIPROCESSING=0 to keep EngineCore in the same
# process, and --enforce-eager to disable CUDA graphs (which cause nsys
# 2023.4.4 import failures due to out-of-order event timestamps).
#
# Note: --enforce-eager may slightly change kernel execution patterns
# compared to CUDA graph mode, but the kernel *types* and block counts
# remain the same — just without graph replay overhead.

GPU_ID=0
VLLM_PORT=8100
MODEL="Qwen/Qwen3-30B-A3B"
GEN_CONCURRENCY=${1:-1}
MAX_NUM_SEQS=${2:-256}
GEN_MAX_TOKENS=512
GPU_MEM_UTIL=0.85

# Timing: vLLM startup ~65s, then warmup 15s, then we want to capture
# steady-state decode. Total delay ~90s before capture starts.
NSYS_DELAY=90
NSYS_CAPTURE=15

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv/bin/activate"
RESULTS_DIR="$DIR/results/nsys_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

source "$VENV"

echo "=== nsys Kernel Trace Capture ==="
echo "nsys delay: ${NSYS_DELAY}s, capture: ${NSYS_CAPTURE}s"
echo "Results: $RESULTS_DIR"
echo ""

VLLM_PID=""
BENCH_PID=""

cleanup() {
    echo "Cleaning up..."
    [ -n "${BENCH_PID:-}" ] && kill "$BENCH_PID" 2>/dev/null || true
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_vllm() {
    echo "Waiting for vLLM..."
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

# ============================================================
# Start vLLM wrapped in nsys with delayed capture
# ============================================================
echo "Starting vLLM under nsys (delay=${NSYS_DELAY}s, capture=${NSYS_CAPTURE}s)..."
echo "  (enforce-eager mode to avoid CUDA graph nsys import bugs)"
echo "  concurrency=$GEN_CONCURRENCY"

# Export so forked child processes (EngineCore_DP0) inherit these
export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_ENABLE_V1_MULTIPROCESSING=0

nsys profile \
    --trace=cuda \
    --force-overwrite=true \
    --delay="$NSYS_DELAY" \
    --duration="$NSYS_CAPTURE" \
    --output="$RESULTS_DIR/gen_profile" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$VLLM_PORT" \
    --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len 4096 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --quantization fp8 \
    --enforce-eager \
    --disable-log-requests \
    > "$RESULTS_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
wait_for_vllm

# Warm up thoroughly
echo "Warming up gen model (15s)..."
python3 "$DIR/gen_benchmark.py" --base-url "http://localhost:$VLLM_PORT" \
    --duration 15 --concurrency $GEN_CONCURRENCY --max-tokens 64 --model "$MODEL" > /dev/null 2>&1

# Run gen benchmark in background - it may crash when nsys stops vLLM
BENCH_DURATION=$((NSYS_DELAY + NSYS_CAPTURE + 30))
echo ""
echo "Running gen benchmark for ${BENCH_DURATION}s (nsys captures from ~${NSYS_DELAY}s)..."
python3 "$DIR/gen_benchmark.py" \
    --base-url "http://localhost:$VLLM_PORT" \
    --duration "$BENCH_DURATION" \
    --concurrency "$GEN_CONCURRENCY" \
    --max-tokens "$GEN_MAX_TOKENS" \
    --model "$MODEL" \
    > "$RESULTS_DIR/gen_output.txt" 2>&1 &
BENCH_PID=$!

# Wait for nsys to finish (vLLM process will terminate after capture)
echo "Waiting for vLLM/nsys to complete..."
wait "$VLLM_PID" 2>/dev/null || true
VLLM_PID=""
echo "vLLM/nsys done."

# Kill benchmark if still running
kill "$BENCH_PID" 2>/dev/null || true
wait "$BENCH_PID" 2>/dev/null || true
BENCH_PID=""
sleep 2

# Show benchmark output
echo ""
echo "Benchmark output:"
cat "$RESULTS_DIR/gen_output.txt" 2>/dev/null || echo "(no output)"

# Check for profile
echo ""
if [ -f "$RESULTS_DIR/gen_profile.nsys-rep" ]; then
    echo "nsys profile: $(ls -lh "$RESULTS_DIR/gen_profile.nsys-rep" | awk '{print $5}')"

    echo ""
    echo "=== Kernel Analysis ==="
    python3 -c "
import sys
sys.path.insert(0, '$DIR')
from analyze_nsys import read_gpu_trace, kernel_duration_stats, theoretical_occupancy, kernel_category_breakdown
import numpy as np
from collections import defaultdict

kernels = read_gpu_trace('$RESULTS_DIR/gen_profile.nsys-rep')
if not kernels:
    print('No kernels found!')
    sys.exit(0)

print(f'Total kernels: {len(kernels)}')
stats = kernel_duration_stats(kernels)
print(f'Kernel duration: mean={stats[\"mean_us\"]:.1f}us  median={stats[\"median_us\"]:.1f}us  p99={stats[\"p99_us\"]:.1f}us')
print(f'Total kernel time: {stats[\"total_ms\"]:.1f}ms')

occ = theoretical_occupancy(kernels, num_sms=132)
print(f'Theoretical SM occupancy: {occ[\"mean_sm_frac\"]*100:.1f}% (mean {occ[\"mean_blocks\"]:.0f} blocks/kernel)')

# Block distribution
blocks = np.array([k['blocks'] for k in kernels])
print(f'')
print(f'Block count distribution:')
print(f'  min: {blocks.min()}  p25: {np.percentile(blocks, 25):.0f}  median: {np.median(blocks):.0f}  p75: {np.percentile(blocks, 75):.0f}  max: {blocks.max()}')
print(f'  Kernels with >=132 blocks (full SM): {(blocks >= 132).sum()} ({(blocks >= 132).sum()/len(blocks)*100:.1f}%)')
print(f'  Kernels with <132 blocks (underutilizing SMs): {(blocks < 132).sum()} ({(blocks < 132).sum()/len(blocks)*100:.1f}%)')
print(f'  Kernels with <66 blocks (<50% SMs): {(blocks < 66).sum()} ({(blocks < 66).sum()/len(blocks)*100:.1f}%)')
print(f'  Kernels with <33 blocks (<25% SMs): {(blocks < 33).sum()} ({(blocks < 33).sum()/len(blocks)*100:.1f}%)')
print(f'  Kernels with 1 block: {(blocks == 1).sum()} ({(blocks == 1).sum()/len(blocks)*100:.1f}%)')

# Duration-weighted SM occupancy
total_time = sum(k['duration'] for k in kernels)
weighted_occ = sum(min(k['blocks'], 132)/132 * k['duration'] for k in kernels) / total_time
print(f'')
print(f'Duration-weighted SM occupancy: {weighted_occ*100:.1f}%')
print(f'  (This accounts for how long each kernel runs, not just count)')

# Time in low-occupancy kernels
low_occ_time = sum(k['duration'] for k in kernels if k['blocks'] < 66)
print(f'  Time in <50% SM kernels: {low_occ_time/1e6:.1f}ms ({low_occ_time/total_time*100:.1f}%)')
very_low_time = sum(k['duration'] for k in kernels if k['blocks'] < 33)
print(f'  Time in <25% SM kernels: {very_low_time/1e6:.1f}ms ({very_low_time/total_time*100:.1f}%)')

# Category breakdown
print(f'')
print(f'Kernel category breakdown:')
cats = kernel_category_breakdown(kernels)
for cat, data in cats.items():
    print(f'  {cat:12s}: {data[\"count\"]:6d} kernels, {data[\"total_ms\"]:8.1f}ms ({data[\"pct\"]:5.1f}%)')

# Top kernel names by total time
name_time = defaultdict(lambda: {'count': 0, 'total_ns': 0, 'blocks': []})
for k in kernels:
    short = k['name'][:120] if k['name'] else '(unnamed)'
    name_time[short]['count'] += 1
    name_time[short]['total_ns'] += k['duration']
    name_time[short]['blocks'].append(k['blocks'])

print(f'')
print(f'Top 20 kernels by total time:')
sorted_names = sorted(name_time.items(), key=lambda x: -x[1]['total_ns'])
total_ns = sum(v['total_ns'] for v in name_time.values())
for name, data in sorted_names[:20]:
    pct = data['total_ns'] / total_ns * 100
    mean_us = data['total_ns'] / data['count'] / 1000
    mean_blocks = np.mean(data['blocks'])
    print(f'  {pct:5.1f}% ({data[\"count\"]:5d}x, mean {mean_us:7.1f}us, {mean_blocks:6.0f} blks) {name}')

# Kernel gap analysis — time between kernels
if len(kernels) > 1:
    gaps = []
    for i in range(1, len(kernels)):
        gap = kernels[i]['start'] - kernels[i-1]['end']
        if gap > 0:
            gaps.append(gap)
    if gaps:
        gaps = np.array(gaps)
        print(f'')
        print(f'Kernel gap analysis (idle time between consecutive kernels):')
        print(f'  Total gaps: {len(gaps)}')
        print(f'  Gap duration: mean={np.mean(gaps)/1000:.1f}us  median={np.median(gaps)/1000:.1f}us  p99={np.percentile(gaps, 99)/1000:.1f}us')
        print(f'  Total gap time: {np.sum(gaps)/1e6:.1f}ms')
        total_span = kernels[-1]['end'] - kernels[0]['start']
        print(f'  Total trace span: {total_span/1e6:.1f}ms')
        print(f'  GPU idle fraction: {np.sum(gaps)/total_span*100:.1f}%')
" 2>&1 | tee "$RESULTS_DIR/kernel_analysis.txt"

else
    echo "WARNING: nsys profile not found"
    ls -la "$RESULTS_DIR/"
    # Check if qdstrm exists (nsys import bug)
    if [ -f "$RESULTS_DIR/gen_profile.qdstrm" ]; then
        echo ""
        echo "Found .qdstrm file (nsys import failed - known bug with CUDA graphs)."
        echo "Size: $(ls -lh "$RESULTS_DIR/gen_profile.qdstrm" | awk '{print $5}')"
    fi
fi

echo ""
echo "Full results in: $RESULTS_DIR"
