# Co-scheduling Generative and Embedding Models on a Single GPU via CUDA MPS

## 1. The Idea

LLM decode is memory-bandwidth-bound. Each decode step reads the full model weights from HBM to generate one token per sequence. The arithmetic intensity is low: the GPU's SMs spend most of their time waiting on memory. This is especially true for Mixture-of-Experts (MoE) models, which load expert weights from HBM but only compute through a small active subset (e.g. 8 of 128 experts per token).

Embedding inference is compute-bound. It runs dense matrix multiplications over full input sequences in a single forward pass, with high arithmetic intensity and high SM utilization per kernel.

These two workloads have complementary resource profiles. If we can run them simultaneously, the embedding model could use SM cycles that the generative model leaves idle, without competing for the same bottleneck resource.

## 2. The Deployment Question

In production, embedding and generative models are typically served on separate GPUs. This is simple but wasteful: the generative GPU's compute units sit partially idle during decode, and the embedding GPU may be underutilized between bursts of indexing traffic.

An alternative: co-locate both models on the same hardware. But on a single GPU, this is strictly worse than having two separate GPUs -- you're splitting one GPU's resources between two workloads, and each gets less than it would standalone. There's no free lunch on 1 GPU.

The interesting case is **two GPUs with TP=2 for both models**. Here the baseline is meaningful:

- **Baseline**: 2 GPUs, gen on GPU 0 (TP=1), embed on GPU 1 (TP=1)
- **Experiment**: 2 GPUs, gen across both (TP=2) + embed across both (TP=2) via MPS

With TP=2, gen gets 2x memory bandwidth (9.6 TB/s vs 4.8 TB/s) and 2x SMs, directly helping the bandwidth-bound MoE decode. (VRAM for KV cache doesn't double -- each GPU still holds both models' weight shards.) If the per-GPU MPS interference from embed is small enough, the TP=2 gen throughput could exceed the TP=1 baseline -- a net win on gen *and* embed from the same 2 GPUs.

The single-GPU experiments below are a proof of concept: measuring MPS interference at peak gen throughput, and comparing MPS against the alternative isolation strategies (MIG and time-slicing).

### VRAM budget (single GPU)

| Component | VRAM |
|---|---|
| Qwen3-30B-A3B-FP8 (FP8 weights) | ~30 GB |
| KV cache (gpu_mem_util=0.50) | ~30 GB |
| Qwen3-Embedding-8B (BF16 weights) | ~17 GB |
| **Total** | **~77 GB / 141 GB** |

## 3. Proof of Concept

### Hardware

- NVIDIA H200 SXM (141 GB HBM3e, 4.8 TB/s bandwidth, 700W TDP, 132 SMs)

### Models

- **Generative**: Qwen/Qwen3-30B-A3B-FP8 (MoE, 128 experts, 8 active/token, pre-quantized FP8, served via vLLM v0.15.1)
- **Embedding**: Qwen/Qwen3-Embedding-8B (dense, BF16, served via vLLM v0.15.1 with `--convert embed`)

### Setup

Both models served via `vllm/vllm-openai:v0.15.1` Docker containers. All experiments use the same configuration:

- Gen: `--gpu-memory-utilization 0.50 --max-model-len 4096 --max-num-seqs 512`
- Embed: `--gpu-memory-utilization 0.30 --max-model-len 512 --convert embed`
- Benchmark: 30s KV-cache warmup, 60s measurement
- Gen load: C=512 concurrent requests, max_tokens=512
- Embed load: C=64 concurrent requests, batch_size=8

**Concurrency tuning**: KV cache capacity at `gpu_mem_util=0.50` is ~322K tokens. At C=512 with ~542 tokens/request (30 input + 512 output), peak demand is ~278K tokens, fitting comfortably without queuing. Higher concurrency (e.g. C=1024) overflows the KV cache, causing request queuing and artificially depressing throughput. C=512 was selected as the point of maximum throughput via sweep.

For MPS, a sidecar Docker container runs the MPS daemon with a shared Docker volume for the pipe directory. All containers use `--ipc=host` for shared memory access.

### 3.1 Co-location via MPS (shared GPU)

Both models run on the full GPU. CUDA MPS enables true parallel kernel execution by routing both processes' work through a shared CUDA context, avoiding time-slicing overhead.

| Condition | Gen (tok/s) | Gen delta | Embed (seq/s) |
|---|---|---|---|
| Gen only | 24,363 | baseline | - |
| Gen + Embed (MPS) | 20,345 | **-16.5%** | 515 |
| Gen + Embed (no MPS, time-sliced) | 22,473 | **-7.8%** | 507 |

**MPS co-location has a measurable cost.** Gen throughput drops 16.5% from 24,363 to 20,345 tok/s when sharing the GPU with the embedding model via MPS. The embedding model achieves 515 seq/s.

**Time-slicing is cheaper than MPS.** Without MPS, CUDA time-slices between the two processes, costing only 7.8% gen throughput. This is the opposite of what MPS documentation would predict: at high GPU utilization, parallel kernel execution (MPS) causes more SM contention than serialized execution (time-slicing), because the embedding model's compute-heavy kernels compete directly with decode for SM resources.

#### Reproducibility (A/B test, 3 reps each)

The MPS vs time-slicing gap is reproducible with tight variance:

| Mode | Rep 1 | Rep 2 | Rep 3 | Mean | Embed mean |
|---|---|---|---|---|---|
| Time-sliced gen | 24,101 | 24,533 | 24,247 | **24,294** | 492 |
| MPS gen | 22,956 | 22,881 | 23,173 | **23,003** | 486 |

MPS is consistently ~1,300 tok/s slower (**-5.3%** gap, ±0.6% across reps). The first-run table above (16.5% gap) included warm-up effects; the stabilized A/B result is the more reliable number.

#### Per-token decode latency

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| Gen only | 20.8ms | 21.6ms | 21.6ms |
| Gen + Embed (MPS) | 23.8ms | 24.5ms | 24.6ms |
| Gen + Embed (no MPS) | 22.7ms | 24.3ms | 24.3ms |

MPS adds ~3ms at p50 (14% increase). Time-slicing adds ~2ms (10% increase).

### 3.2 MIG baseline (partitioned GPU)

NVIDIA Multi-Instance GPU (MIG) provides hardware-level isolation by partitioning a GPU into fixed slices with dedicated SMs, memory controllers, and cache. This is the "hard isolation" alternative to MPS's flexible resource sharing.

**Partition layout:** 4g.71gb (4/7 SMs, 4/8 memory) for gen + 3g.71gb (3/7 SMs, 4/8 memory) for embed. This uses all 7/7 SM slices. Both partitions get ~70 GB VRAM.

MIG partitions are passed to Docker containers via `--gpus "device=MIG-<uuid>"`. Each container sees its partition as a plain GPU. Gen uses `--gpu-memory-utilization 0.85` to fit the 30 GB model in the 70 GB partition.

| Condition | Gen (tok/s) | Gen delta | Embed (seq/s) |
|---|---|---|---|
| Gen only (4g.71gb partition) | 4,885 | -79.9% vs full | - |
| Gen + Embed (MIG isolated) | 4,901 | -79.9% vs full | 513 |

As expected, MIG gives **perfect isolation**: zero interference between partitions. But the cost is catastrophic for the bandwidth-bound gen model:

**Gen throughput drops 80% vs the full GPU** (4,885 vs 24,363 tok/s). The 4g partition has 4/8 = 50% of the GPU's memory bandwidth and 4/7 = 57% of SMs, but gen only achieves 20% of full-GPU throughput. The reduced SM count limits how many concurrent decode operations can run, preventing the workload from saturating even the reduced bandwidth.

**Embed throughput is identical across all conditions** (~507-515 seq/s across MPS, time-sliced, and MIG). At C=64 the embedding workload is bottlenecked by vLLM scheduling overhead rather than GPU resources.

#### Per-token decode latency (MIG)

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| MIG gen only (4g) | 104.6ms | 105.5ms | 105.5ms |
| MIG gen + embed (4g+3g) | 104.2ms | 105.4ms | 105.4ms |

MIG partitioning 5x per-token latency (105ms vs 21ms) due to the reduced compute capacity.

### 3.3 Profiling: Why is MPS slower than time-slicing?

#### DCGM aggregate metrics

`dcgmi dmon` captured SM Active, SM Occupancy, Tensor Active, DRAM Active, FP32 Active, and FP16 Active at 1-second intervals across gen-only, time-sliced, and MPS conditions (60 samples each):

| Metric | Gen Only | Time-sliced | MPS |
|---|---|---|---|
| SM Active | 0.836 | 0.845 | 0.856 |
| SM Occupancy | 0.180 | 0.179 | 0.181 |
| Tensor Active | 0.178 | 0.203 | 0.199 |
| DRAM Active | 0.301 | 0.318 | 0.306 |

All three conditions are nearly identical at the aggregate level. DCGM cannot distinguish the mechanism — the 5% throughput difference happens below its sampling resolution.

#### ncu (Nsight Compute) kernel-level L2 cache analysis

**MPS limitation:** ncu does not support profiling with CUDA MPS enabled (`Profiling is not supported with Multi-Process Server (MPS) enabled`). The co-located comparison uses time-slicing instead. Since both MPS and time-slicing share the same physical L2 cache, L2 contention analysis is valid for either mode.

Kernels profiled with `MemoryWorkloadAnalysis` section (`--launch-skip 20000 --launch-count 3`):

**`fused_add_rms_norm_kernel`** (directly comparable — same kernel in both runs):

| Metric | Gen Only | Co-located (time-sliced) |
|---|---|---|
| **L2 Hit Rate** | **53.60%** | **53.04%** |
| L1/TEX Hit Rate | 84.22% | 84.22% |
| Memory Throughput | 768 GB/s | 858 GB/s |
| Mem Busy | 21.95% | 20.49% |

L2 hit rate drops by only 0.56 percentage points with co-location — **L2 cache contention is not the mechanism** causing the throughput difference. The H200's 51 MB L2 is large enough to accommodate both workloads without meaningful eviction.

**Other kernels profiled** (gen-only run):

| Kernel | L2 Hit Rate | Mem Throughput | L1/TEX Hit |
|---|---|---|---|
| `deep_gemm::fp8_gemm_kernel` (MoE expert GEMM) | 60.48% | 1,483 GB/s | 46.94% |
| `finalizeMoeRoutingKernel` | 16.05% | 1,090 GB/s | 75.33% |
| `fused_add_rms_norm_kernel` | 53.60% | 768 GB/s | 84.22% |

The MoE expert GEMM is the most bandwidth-intensive kernel (1.5 TB/s, 30% of peak) with a moderate L2 hit rate. The routing kernel has the worst L2 hit rate (16%) due to scattered expert selection patterns.

#### nsys (Nsight Systems) kernel dispatch timing

To test whether MPS adds per-kernel dispatch overhead, nsys traces were captured for two conditions: gen-only without MPS, and gen-only with the MPS daemon running (no embed process). Both conditions run exactly the same workload (64 prompts × 128 tokens via `ncu_decode_probe.py`), so any timing difference is attributable to MPS routing.

**CUDA API launch latency (CPU-side):**

| API Call | No-MPS Avg (ns) | MPS Avg (ns) | Delta |
|---|---|---|---|
| `cudaLaunchKernel` (21,584 calls) | 43,675 | 43,473 | **-0.5%** |
| `cuLaunchKernelEx` (9,093 calls) | 37,204 | 37,285 | **+0.2%** |
| `cudaLaunchKernelExC` (6,137 calls) | 7,903 | 7,737 | **-2.1%** |
| `cuLaunchKernel` (924 calls) | 4,676 | 4,708 | **+0.7%** |
| `cudaMemcpyAsync` (39,558 calls) | 122,236 | 119,816 | **-2.0%** |

All launch APIs are within noise (<2.1%). **MPS does not add measurable per-kernel dispatch latency.**

**GPU kernel execution time (top kernels by total time):**

| Kernel | No-MPS Avg (ns) | MPS Avg (ns) | Delta |
|---|---|---|---|
| `sm90_fp8_gemm_1d2d_impl` (MoE expert GEMM) | 763,808 | 763,835 | **+0.004%** |
| `delayStreamKernel` | 1,024,502 | 1,024,492 | **-0.001%** |
| `fp8_blockscale_gemm` | 104,937 | 105,012 | **+0.07%** |
| `rms_norm_kernel` | 21,284 | 21,568 | **+1.3%** |
| `fused_add_rms_norm_kernel` | 9,142 | 9,324 | **+2.0%** |
| `topkGating` | 6,424 | 6,680 | **+4.0%** |

Large kernels (>100us) are identical. Small kernels (6-22us) show up to 4% variation, consistent with normal run-to-run noise rather than systematic MPS overhead.

**Aggregate GPU time:**

| Metric | No-MPS | MPS | Delta |
|---|---|---|---|
| Total kernel GPU time | 2,470.88 ms | 2,475.93 ms | **+0.20%** |
| Average kernel duration | 65.47 us | 65.60 us | **+0.20%** |
| Steady-state GPU utilization | 39.06% | 39.13% | **+0.18%** |

#### nsys concurrent test (gen + active embed)

To test whether concurrent embed load slows down gen kernels via bandwidth competition, a second nsys test captured gen kernel timing under three conditions: (1) gen-only baseline, (2) gen with active embed via MPS, (3) gen with active embed via time-slicing. The embed server was receiving C=64 concurrent requests throughout.

**Top kernel comparison (`sm90_fp8_gemm_1d2d_impl` 1536×2048, MoE expert GEMM):**

| Condition | Avg (ns) | Delta vs baseline |
|---|---|---|
| gen_only | 763,437 | baseline |
| mps_active (gen + embed concurrent) | 763,567 | **+0.02%** |
| ts_active (gen + embed concurrent) | 763,831 | **+0.05%** |

**Batch throughput (64 prompts × 128 tokens):**

| Condition | Output tok/s |
|---|---|
| gen_only | 641.7 |
| mps_active | 647.8 |
| ts_active | 650.7 |

All three conditions produce **identical kernel timing and batch throughput**, even with the embed model actively processing concurrent requests. The concurrent embed load does not measurably slow down individual gen kernels.

#### Root cause: CUDA graph replay interleaving

The nsys and offline batch tests all use `enforce_eager=True` (no CUDA graphs), while the vLLM server uses CUDA graphs by default. Testing both modes reveals CUDA graphs as the mechanism:

| Condition | CUDA Graphs | Throughput (tok/s) | Decode p50 |
|---|---|---|---|
| eager gen-only | OFF | 10,668 | 20.9 tok/s/req |
| eager MPS + active embed | OFF | 10,279 (-3.6%) | 20.1 tok/s/req |
| eager time-sliced + active embed | OFF | 9,885 (-7.3%) | 18.8 tok/s/req |
| graphs MPS + active embed | ON | 22,030 | 43.7 tok/s/req |
| graphs time-sliced + active embed | ON | 23,113 | 46.4 tok/s/req |

**Without CUDA graphs**: MPS is 4.0% **faster** than time-slicing (10,279 vs 9,885 tok/s). This is the expected result — MPS avoids context-switch overhead, and individual kernels run at identical speed regardless of sharing mode (confirmed by nsys).

**With CUDA graphs**: MPS is 4.7% **slower** than time-slicing (22,030 vs 23,113 tok/s). This matches the A/B test's 5.3% gap.

**The mechanism**: CUDA graphs pre-record a sequence of kernel launches and replay them as a single GPU operation, eliminating per-kernel launch overhead (~2.2x throughput improvement). Under time-slicing, the graph replays atomically — the entire kernel sequence executes back-to-back without interruption from the embed process. Under MPS, both processes share a single CUDA context, so the embed model's kernels can be **interleaved into the middle of gen's CUDA graph replay**, breaking the back-to-back execution that graphs are designed to provide. Each interleaved embed kernel adds a small gap in the gen graph's execution, and at ~37k kernels per generation cycle, these gaps accumulate to the observed ~5% overhead.

This also explains why the overhead doesn't appear in individual kernel timing (nsys): each gen kernel still runs at full speed, but the **gaps between kernels** within a graph replay are longer because embed kernels are inserted between them.

#### Summary of hypotheses tested

| Hypothesis | Test | Result |
|---|---|---|
| L2 cache contention | ncu L2 hit rate | 0.56 pp drop — **ruled out** |
| MPS dispatch latency | nsys kernel launch timing | Identical — **ruled out** |
| Kernel execution overhead | nsys kernel duration | Identical — **ruled out** |
| Memory bandwidth competition | nsys concurrent test | Identical — **ruled out** |
| CUDA graph interleaving | enforce-eager A/B test | **Confirmed** — gap disappears without graphs, reverses sign |

### 3.4 Comparison

| Configuration | Gen (tok/s) | Gen % of full | Embed (seq/s) | Gen interference |
|---|---|---|---|---|
| Full GPU, gen only | 24,363 | 100% | - | - |
| Full GPU, embed only | - | - | 479 | - |
| Full GPU, MPS co-located | 23,003* | 94.4% | 486* | **-5.6%** |
| Full GPU, time-sliced co-located | 24,294* | 99.7% | 492* | **-0.3%** |
| MIG 4g, gen only | 4,885 | 20.1% | - | -79.9% (partitioning) |
| MIG 3g, embed only | - | - | 490 | - |
| MIG 4g+3g, co-located | 4,901 | 20.1% | 513 | 0% within MIG |

*A/B test means (3 reps each, alternating). Initial single-run values were noisier (MPS: 20,345, time-sliced: 22,473).

**Time-slicing is the best co-location strategy for this workload.** The stabilized A/B test shows time-slicing costs essentially nothing (-0.3% gen, within noise) while providing ~492 embed seq/s. MPS is slightly worse (-5.6%), with ncu profiling ruling out L2 cache contention as the cause (see §3.3). MIG's hard partitioning remains catastrophic for bandwidth-bound workloads.

**Embed throughput is bottlenecked by vLLM overhead, not GPU resources.** Embed achieves ~479-513 seq/s across all conditions (full GPU standalone, MPS, time-sliced, MIG 3g partition), confirming the GPU has ample capacity for this workload regardless of co-location strategy.

The near-zero interference from time-slicing on a single GPU is encouraging for the TP=2 experiment: if per-GPU interference is negligible, TP=2 gen throughput could scale nearly 2x while still serving embeddings.

## 4. TP=2 Experiment

Not yet conducted. This is where the real value proposition lives.

### The baseline

2 GPUs, each dedicated to one workload:
- GPU 0: gen model (TP=1), full 141 GB for weights + KV cache, 4.8 TB/s bandwidth
- GPU 1: embed model (TP=1), using ~15 GB of 141 GB

This is how you'd deploy today. GPU 1 is massively underutilized.

### The experiment

Same 2 GPUs, both workloads on both GPUs via TP=2 + MPS:
- Gen model: TP=2 across both GPUs, ~15 GB weights per GPU, KV cache split across both
- Embed model: TP=2 across both GPUs, ~7.5 GB weights per GPU
- MPS on each GPU enables parallel kernel execution between gen and embed

### Why this should work

The single-GPU A/B test shows time-slicing causes essentially zero gen interference (-0.3%, within noise) and MPS only -5.6%. With TP=2, gen gets 2x memory bandwidth (9.6 TB/s total). Even in the MPS case, gen only needs to scale >1.06x from TP=2 to beat the TP=1 baseline — well within expectations for a bandwidth-bound workload.

From the single-GPU data, time-slicing is the better co-location strategy. The TP=2 experiment should test both, though time-slicing is strongly favored.

### What to measure

| Condition | GPUs | Gen | Embed |
|---|---|---|---|
| Baseline: separate GPUs | 2 | TP=1 on GPU 0 | TP=1 on GPU 1 |
| Co-located TP=2 + MPS | 2 | TP=2 on both | TP=2 on both |
| Co-located TP=2 + time-sliced | 2 | TP=2 on both | TP=2 on both |

Success: co-located gen throughput >= baseline gen throughput, with embed throughput as bonus.

## 5. Conclusion

Not yet written. Pending TP=2 results.
