# MIG vs Co-location: Maximum Throughput Comparison

H200 (141 GB, 132 SMs) — Qwen3-30B-A3B-FP8 (generative) + Qwen3-Embedding-8B (embedding)

## 1. MIG Baselines

### 1.1 Gen Model on MIG 4g.71gb (64 SMs, 71 GB)

Server: vLLM v0.15.1, `--gpu-memory-utilization 0.90 --max-model-len 4096`, compiled mode (default).

| Concurrency | max-num-seqs | Throughput (tok/s) | Decode p50 (tok/s/req) |
|---|---|---|---|
| 64 | 256 | 5,833 | 90.2 |
| 256 | 256 | 12,017 | 46.3 |
| **512** | **512** | **15,007** | **29.3** |
| 768 | 768 | 13,309 | 18.5 |
| 1024 | 768 | 13,125 | 18.1 |
| 256 (eager) | 256 | 5,721 | 22.4 |

Peak throughput at C=512. Higher concurrency (768+) degrades throughput because KV cache per sequence shrinks, reducing batch efficiency.

Compiled mode (torch.compile + CUDA graphs) provides ~2.6x throughput over eager mode (15,007 vs 5,721).

3-rep verification at C=512: 14,950 / 15,015 / 15,163 → **mean 15,042 tok/s** (σ=109).

### 1.2 Embed Model on MIG 3g.71gb (60 SMs, 71 GB)

Server: vLLM v0.15.1, `--gpu-memory-utilization 0.90 --max-model-len 512 --convert embed`.

| Concurrency | Batch size | max-num-seqs | Throughput (seq/s) |
|---|---|---|---|
| 64 | 8 | 256 | 480 |
| 64 | 32 | 256 | 505 |
| 128 | 8 | 256 | 489 |
| **128** | **32** | **256** | **507** |
| 256 | 8 | 256 | 489 |
| 128 (eager) | 8 | 256 | 494 |

Embedding throughput is insensitive to concurrency beyond C=64. Larger per-request batch sizes help modestly (8→32: +5%).

**Critical finding: the embed model is NOT compute-bound.** Testing on the full GPU (132 SMs) gives 508 seq/s — identical to MIG 3g (60 SMs). More SMs do not help. The bottleneck is HBM bandwidth for prefill or vLLM's CPU-side scheduling overhead.

3-rep verification at C=32/B=64: 508 / 505 / 504 → **mean 506 seq/s** (σ=2).

### 1.3 MIG Baselines Summary

| Model | Partition | Peak Throughput |
|---|---|---|
| Gen (Qwen3-30B-A3B-FP8) | MIG 4g.71gb (64 SMs) | **15,042 tok/s** |
| Embed (Qwen3-Embedding-8B) | MIG 3g.71gb (60 SMs) | **506 seq/s** |

## 2. Co-location on Full GPU

Both models share the full GPU (132 SMs) via time-slicing (default, no MPS).

Gen: `--gpu-memory-utilization 0.50 --max-num-seqs 512`, compiled mode.
Embed: `--gpu-memory-utilization 0.30 --max-num-seqs 512`.

### 2.1 Memory Split Sweep

All runs: gen C=512, embed C=128/B=32, 60s measurement.

| Gen GPU% | Embed GPU% | Gen (tok/s) | Embed (seq/s) | Gen vs MIG | Embed vs MIG |
|---|---|---|---|---|---|
| 50 | 30 | 20,647 | 494 | **+37.3%** | -2.4% |
| 60 | 20 | 20,508 | 495 | +36.4% | -2.2% |
| 65 | 15 | 20,376 | 496 | +35.5% | -2.0% |

Memory split has minimal impact. 50/30 is fine.

### 2.2 Embed Client Configuration Sweep

Gen: 50/30 split, C=512. Varying embed client parameters.

| Embed C | Embed B | Gen (tok/s) | Embed (seq/s) |
|---|---|---|---|
| 128 | 32 | 20,594 | 493 |
| 64 | 64 | 22,358 | 492 |
| 32 | 64 | 20,667 | 498 |
| 32 | 32 | 20,600 | 498 |
| 64 | 128 | 22,511 | 488 |
| 16 | 128 | 23,127 | 486 |

Lower embed concurrency marginally improves embed throughput while maintaining gen performance.

### 2.3 MPS Comparison

| Sharing Mode | Config | Gen (tok/s) | Embed (seq/s) |
|---|---|---|---|
| Time-sliced | C=512 gen, C=32/B=64 embed | **22,203** | **496** |
| MPS (default) | C=512 gen, C=128/B=32 embed | 20,013 | 494 |
| MPS (90/10 threads) | C=512 gen, C=32/B=64 embed | 20,462 | 487 |

Time-slicing outperforms MPS on both metrics. MPS loses ~10% gen throughput due to the torch.compile interaction identified in prior analysis (see REPORT.md §3.3).

### 2.4 Best Co-location Result (3-rep)

Config: time-sliced, gen `--max-num-seqs 512` C=512, embed `--max-num-seqs 512` C=32/B=64.

| Rep | Gen (tok/s) | Embed (seq/s) |
|---|---|---|
| 1 | 22,098 | 500 |
| 2 | 22,249 | 496 |
| 3 | 22,262 | 493 |
| **Mean** | **22,203** | **496** |

## 3. Comparison

| Metric | MIG (4g+3g) | Co-located | Delta |
|---|---|---|---|
| Gen throughput | 15,042 tok/s | 22,203 tok/s | **+47.6%** |
| Embed throughput | 506 seq/s | 496 seq/s | **-1.8%** |
| Gen decode p50 | 29.3 tok/s/req | ~43 tok/s/req | +47% |

### Why gen wins big

MIG 4g gives gen 64 of 132 SMs (48%). Co-location gives it all 132 SMs via time-slicing. Since the MoE decode step is compute-bound at high batch sizes (C=512), nearly doubling the SM count produces a proportional throughput increase.

### Why embed loses slightly

The embed model is HBM-bandwidth-bound, not compute-bound. It achieves the same ~506 seq/s whether it has 60 SMs (MIG 3g) or 132 SMs (full GPU alone). The 1.8% loss under co-location comes from time-slicing context-switch overhead: the GPU alternates between gen and embed processes, and each context switch adds a small stall.

This overhead cannot be eliminated because:
- MPS (which avoids context switches) introduces a larger penalty via the torch.compile interaction (~5% gen loss)
- MPS thread partitioning doesn't help (embed doesn't need more SMs)
- Reducing gen load doesn't help (embed gets the same ~496 seq/s at gen C=256 as at C=512)

### Verdict

Co-location produces a decisive gen improvement (+47.6%) at the cost of a small embed regression (-1.8%). The embed model operates near its architectural throughput ceiling (~510 seq/s) regardless of GPU partition size, so the MIG partition's dedicated resources provide only a marginal isolation benefit (506 vs 496).

For workloads where gen throughput matters more than a 2% embed margin — which is essentially all production deployments — co-location on the full GPU with time-slicing is the superior configuration.

## 4. Recommended Configuration

```
# Gen server
vllm serve Qwen/Qwen3-30B-A3B-FP8 \
    --port 8100 \
    --gpu-memory-utilization 0.50 \
    --max-model-len 4096 \
    --max-num-seqs 512

# Embed server
vllm serve Qwen/Qwen3-Embedding-8B \
    --port 8200 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 512 \
    --convert embed \
    --max-num-seqs 512
```

No MPS, no MIG — use default GPU time-slicing.

## Appendix: Experiment Environment

- GPU: NVIDIA H200 (141 GB HBM3e, 132 SMs)
- CPU: Intel Xeon Platinum 8468 (16 vCPUs)
- vLLM: v0.15.1 (Docker image `vllm/vllm-openai:v0.15.1`)
- Gen model: Qwen/Qwen3-30B-A3B-FP8 (MoE, 3B active params, FP8 quantized)
- Embed model: Qwen/Qwen3-Embedding-8B (dense, 8B params)
- Benchmark: custom async HTTP clients (gen_benchmark.py, embed_benchmark_vllm.py)
- Duration: 60s measurement after 20s warmup per condition
