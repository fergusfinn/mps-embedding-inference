#!/bin/bash
set -euo pipefail

# Capture nsys traces for gen decode with and without MPS
# to measure per-kernel dispatch latency differences.

VLLM_IMAGE="vllm/vllm-openai:v0.15.1"
HF_CACHE="/home/ubuntu/.cache/huggingface"
VLLM_CACHE="/home/ubuntu/.cache/vllm"
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$DIR/profile_results"
mkdir -p "$RESULTS"

full_cleanup() {
    docker stop nsys-gen mps-daemon 2>/dev/null || true
    docker rm nsys-gen mps-daemon 2>/dev/null || true
    docker volume rm mps-pipe 2>/dev/null || true
}
trap full_cleanup EXIT

echo "=== nsys kernel timing: no-MPS vs MPS ==="

# --- Run 1: No MPS ---
echo ""
echo "--- Run 1: gen decode WITHOUT MPS ---"
full_cleanup
sleep 2

NSYS=/opt/nsys/bin/nsys

docker run --rm --name nsys-gen --gpus all --network host --ipc=host --privileged \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -v "$VLLM_CACHE:/root/.cache/vllm" \
    -v "$DIR:/workspace" \
    -v /opt/nvidia/nsight-systems/2024.6.2:/opt/nsys \
    -e HF_HUB_OFFLINE=1 \
    --entrypoint bash \
    "$VLLM_IMAGE" \
    -c "$NSYS profile --trace=cuda -o /workspace/profile_results/nsys_no_mps --force-overwrite true -- python3 /workspace/ncu_decode_probe.py 2>&1 && echo 'nsys done'"

echo "  No-MPS trace captured"

# --- Run 2: With MPS ---
echo ""
echo "--- Run 2: gen decode WITH MPS ---"
full_cleanup
sleep 2

docker volume create mps-pipe > /dev/null 2>&1 || true
docker run --rm -d --name mps-daemon \
    --gpus all --ipc=host --entrypoint bash \
    -v mps-pipe:/mps "$VLLM_IMAGE" \
    -c 'CUDA_MPS_PIPE_DIRECTORY=/mps nvidia-cuda-mps-control -d && sleep infinity' \
    > /dev/null 2>&1
sleep 2

docker run --rm --name nsys-gen --gpus all --network host --ipc=host --privileged \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -v "$VLLM_CACHE:/root/.cache/vllm" \
    -v "$DIR:/workspace" \
    -v /opt/nvidia/nsight-systems/2024.6.2:/opt/nsys \
    -v mps-pipe:/mps -e CUDA_MPS_PIPE_DIRECTORY=/mps \
    -e HF_HUB_OFFLINE=1 \
    --entrypoint bash \
    "$VLLM_IMAGE" \
    -c "$NSYS profile --trace=cuda -o /workspace/profile_results/nsys_with_mps --force-overwrite true -- python3 /workspace/ncu_decode_probe.py 2>&1 && echo 'nsys done'"

echo "  MPS trace captured"

# --- Extract stats ---
echo ""
echo "--- Extracting kernel stats ---"

NSYS_HOST=/opt/nvidia/nsight-systems/2024.6.2/bin/nsys

for label in no_mps with_mps; do
    echo ""
    echo "=== $label: kernel summary ==="
    $NSYS_HOST stats --report cuda_gpu_kern_sum "$RESULTS/nsys_${label}.nsys-rep" 2>/dev/null || echo "nsys stats failed for $label"
    echo ""
    echo "=== $label: cuda api sum ==="
    $NSYS_HOST stats --report cuda_api_sum "$RESULTS/nsys_${label}.nsys-rep" 2>/dev/null || echo "nsys api stats failed for $label"
done

echo ""
echo "=== Done ==="
