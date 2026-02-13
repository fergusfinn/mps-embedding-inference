# MPS for Co-located Embedding + Generative Inference

## What we're trying to prove

That co-locating a compute-bound embedding model with a memory-bandwidth-bound generative model on the same GPU(s) yields higher aggregate throughput per GPU than running each workload on dedicated hardware, by exploiting the complementary resource profiles of the two workloads.

## Why we think it might work

### The core observation

During autoregressive decode, generative LLMs are memory-bandwidth-bound. Each token requires streaming the full model weights from HBM to perform what amounts to a matrix-vector product. The GPU's SMs are largely idle — they finish the arithmetic faster than the memory subsystem can feed them data.

Embedding models are compute-bound. They run a full forward pass over the input sequence in a single shot — dense matrix multiplications that keep the SMs and Tensor Cores busy.

These two workload profiles are complementary: one starves for compute, the other starves for memory bandwidth. On a single GPU, the hardware scheduler can interleave warps from both workloads, filling SMs with embedding compute while the generative model waits on HBM.

### Co-scheduling mechanisms

The core hypothesis requires concurrent kernel execution from both workloads. There are several ways to achieve this, with different tradeoffs:

| Mechanism | Kernel overlap? | Code changes to vLLM/serving stack? | Operational complexity | Notes |
|-----------|----------------|--------------------------------------|----------------------|-------|
| **MPS** | Yes — single CUDA context, hardware scheduler interleaves | None — just start the MPS daemon, run two unmodified processes | Low (daemon setup) | MPS daemon is a SPOF; GPU fault in either process takes down both |
| **Two processes (time-sliced)** | No — context switches (~25-50us), one process at a time | None | None | Baseline for comparison; no overlap, just switching overhead |
| **MIG** | No — hard SM/memory partitioning | None | Low | Guarantees isolation but prevents SM scavenging; defeats the purpose |
| **libsmctrl** | Yes — software SM partitioning with dynamic allocation | Yes — significant integration work | High (research tool) | Most fine-grained control; could dynamically shift SMs between workloads based on decode vs prefill phase |

MPS is the natural starting point: it enables kernel overlap with zero code changes to the serving stack. The single-GPU experiment should compare MPS vs time-sliced to validate that overlap is actually happening.

## Assumptions

- **Throughput-oriented workloads with starvation avoidance.** We're optimizing for aggregate throughput, not latency. Co-scheduling will introduce jitter — that's fine. What matters is that neither workload gets starved: both must meet long but concrete SLOs (e.g. "every embedding request completes within X seconds", "generative decode doesn't stall for more than Y ms").
- **HBM capacity.** Both models' weights, the generative model's KV cache, and embedding batch activations must fit in GPU memory simultaneously. This is a hard constraint, not an assumption — the experiments must be designed around it.

## Proposed experiment

### Preliminary: single-GPU SM scavenging test

A single-GPU test that answers a narrower question: **how much free embedding throughput can you extract from a GPU already running generative decode, without hurting it?**

The generative model's configuration is held constant across all steps, and we measure whether MPS lets us scavenge idle SMs.

**Step 1 — Generative baseline:** Run the generative model alone at a fixed batch size. Record tokens/sec and VRAM usage. Note how much VRAM remains free.

**Step 2 — Co-located with MPS:** Same generative config, same batch size, same KV cache budget. Launch the embedding model as a second process via MPS, sized to fit in whatever VRAM is left over. Measure:
- Generative tokens/sec (should be ~unchanged if the hypothesis holds)
- Embedding sequences/sec (this is "free" throughput from otherwise-idle SMs)

**Step 3 — Co-located without MPS (time-sliced):** Same as step 2, but without MPS. Both processes context-switch on the GPU. Expect both workloads to degrade vs the MPS case.

**Observability:** Capture `nsys` profiles for each step. Key things to look for:
- **Kernel timeline:** Are embedding kernels actually executing concurrently with decode kernels (step 2), or are they serialized? This is the smoking gun for whether MPS overlap is real.
- **SM occupancy over time:** How much of the GPU's compute capacity is idle during decode-only (step 1) vs co-located (step 2)?
- **Memory bandwidth utilization:** Is HBM bandwidth still saturated during decode when embedding kernels are co-running, or does the embedding workload contend for bandwidth too?
- **Kernel durations:** Do individual decode kernels take longer when embedding is co-located? Even if throughput is maintained, slower kernels could indicate contention at the memory controller or L2 cache.

Install `nsys` from the CUDA toolkit — it works on consumer GPUs (4090s). DCGM (`dcgmi`) is data center only and won't work here.

**What a good result looks like:**
- Step 2 generative throughput is within ~5% of step 1 (embedding work didn't steal from decode)
- Step 2 embedding throughput is meaningfully non-zero (idle SMs were put to work)
- Step 3 shows degradation in both workloads vs step 2 (MPS is doing something useful vs time-slicing)
- `nsys` timeline shows actual kernel overlap in step 2, not just interleaving

If step 2 shows significant generative degradation, the core hypothesis is wrong and there's no point running the 2-GPU experiment.

#### Run 1 results: Qwen2.5-3B-Instruct + e5-large-v2 on RTX 4090

| Metric | Step 1 (gen only) | Step 2 (MPS) | Step 3 (time-sliced) |
|--------|-------------------|--------------|----------------------|
| Gen throughput (tok/s) | **869** | 238–354 (59–73% drop) | 369–481 (45–58% drop) |
| Embed throughput (seq/s) | — | 420–433 | 281–331 |
| SM utilization | 90–100% | 100% | 100% |
| Memory bandwidth util | 81–93% | 45% (throttled) | 50–53% (throttled) |
| GPU clock (pclk MHz) | 2745–2760 | 2310–2370 | 2685–2730 |

**Findings:**
1. **The core assumption was wrong for this config.** Qwen 3B decode already saturates SMs at ~93-100% utilization on a 4090 with batch_size=8. There are no idle SMs to scavenge.
2. **Power/thermal throttling.** Adding the embedding workload pushed total power draw past the GPU's limit. The GPU reduced clocks from ~2760 to ~2325 MHz (MPS) / ~2700 MHz (time-sliced), which halved memory bandwidth utilization and destroyed generative throughput.
3. **MPS was worse than time-slicing.** True kernel overlap via MPS caused more interference (both workloads fighting for SMs and memory bandwidth simultaneously) than time-slicing (which at least gives each workload exclusive access during its time slice).

**Conclusion:** The hypothesis requires a model/config where decode genuinely leaves SMs idle — i.e., a model that is more clearly memory-bandwidth-bound. Qwen 3B on a 4090 is not that. Re-running with Llama 3.1 8B (2.5x larger, proportionally more bandwidth per token, fewer SMs active per decode step) and batch_size=1 to maximize the bandwidth-bound regime.

#### Run 2 results: Llama 3.1 8B + e5-large-v2 on RTX 4090 (with nsys kernel tracing)

| Metric | Step 1 (gen only) | Step 2 (MPS) | Step 3 (time-sliced) |
|--------|-------------------|--------------|----------------------|
| Gen throughput (tok/s) | **61.4** | 24.7 (60% drop) | 28.2 (54% drop) |
| Embed throughput (seq/s) | — | 364.7 | 300.4 |
| SM utilization (dmon) | 100% | 100% | 100% |
| Memory bandwidth util (dmon) | 100% | 60% (throttled) | 64–67% (throttled) |
| GPU clock (pclk MHz) | 2745 | 2475–2505 | 1793–2730 (oscillating) |
| Gen kernels (nsys) | 41,605 | 21,260 | 21,260 |
| Gen kernel mean duration | 60.5 us | 65.5 us (+8.3%) | 65.4 us (+8.1%) |
| Gen theoretical SM occupancy | 77.2% (935 blocks/kernel) | 82.1% (922 blocks/kernel) | 82.1% (922 blocks/kernel) |
| Embed theoretical SM occupancy | — | 98.9% (3120 blocks/kernel) | 98.9% (3120 blocks/kernel) |
| Kernel overlap (nsys) | — | 1165.8ms (83.7% of gen time) | 1344.0ms (96.7% of gen time) |

**Findings:**
1. **Same pattern as Run 1 — core assumption still wrong.** Even with a 2.5x larger model at concurrency=1, Llama 8B decode uses ~77% of 128 SMs on the 4090. There are no idle SMs to scavenge.
2. **nsys confirmed kernel overlap is real.** Under MPS, 83.7% of generative kernel time overlapped with embedding kernels. Under time-slicing, 96.7% overlapped (surprisingly — time-slicing on the 4090 doesn't fully serialize CUDA contexts).
3. **Power throttling is the killer.** GPU clocks dropped from 2745 MHz to ~2490 MHz (MPS) and oscillated between 1793–2730 MHz (time-sliced). Memory bandwidth utilization fell from 100% to ~60%. The throughput drop comes from power-limited clock reduction, not from SM contention per se.
4. **Individual gen kernels barely slowed down** (+8% under both MPS and time-sliced). The throughput collapse comes from fewer kernels executing in the same wall time due to lower clocks, not from individual kernels taking longer.
5. **MPS still worse than time-slicing for gen** (24.7 vs 28.2 tok/s), but better for embed (364.7 vs 300.4 seq/s). Under MPS, concurrent execution means both workloads continuously compete for the power budget. Time-slicing gives each workload brief periods at full clock.

**Conclusion:** The hypothesis is falsified on consumer RTX 4090 hardware. The fundamental problem is **power-limited clock throttling**, not SM scheduling. The 4090's 450W TDP is a shared budget between compute and memory — any additional compute load forces clock reductions that directly hurt the bandwidth-bound generative workload.

#### Run 3 results: Qwen3-30B-A3B MoE (FP8) + Qwen3-Embedding-8B on H200

| Metric | Step 1 (gen only) | Step 2 (MPS) | Step 3 (time-sliced) | Step 4 (embed only) |
|--------|-------------------|--------------|----------------------|---------------------|
| Gen throughput (tok/s) | **195.2** | 143.3 (-26.6%) | 78.2 (-59.9%) | — |
| Embed throughput (seq/s) | — | 335.8 | 308.5 | **350.5** |
| SM utilization (dmon) | 100% | 100% | 100% | 62% |
| Memory bandwidth util | 18% | 32% | 24% | 19% |
| GPU clock (pclk MHz) | 1980 | 1980 | 1980 | 1980 |
| Power draw (W) | ~230 | ~395 | ~345 | ~315 |
| Temperature (C) | 37 | 44 | 43 | 43 |
| VRAM used (both models) | 67 GB | 82 GB | 82 GB | 14.5 GB |

**Hardware:** NVIDIA H200 (141 GB HBM3e, 700W TDP, 4.8 TB/s bandwidth, 132 SMs)

**Models:**
- Generative: Qwen3-30B-A3B (BF16 weights, dynamic FP8 quantization via vLLM), ~29 GB on GPU
- Embedding: Qwen3-Embedding-8B (BF16, direct transformers), ~14.4 GB on GPU
- Config: concurrency=1, max_tokens=512, embed batch_size=8, gpu_mem_util=0.45

**Key findings:**

1. **No power throttling!** GPU clocks stayed at 1980 MHz across all steps. Power peaked at ~400W under MPS, well under the 700W TDP. This was the primary failure mode on the 4090 and it's completely eliminated on the H200.

2. **MPS is dramatically better than time-slicing.** Gen throughput under MPS: 143.3 tok/s (-26.6%), time-sliced: 78.2 tok/s (-59.9%). MPS preserves 2.3x more gen throughput than time-slicing. This validates that MPS concurrent kernel execution is meaningfully better than context switching for this workload mix.

3. **Embedding throughput is near-peak under MPS.** 335.8 seq/s under MPS vs 350.5 seq/s standalone = only -4.2% degradation. Under time-slicing: 308.5 seq/s = -12.0% degradation. The embedding model gets nearly all of its standalone performance "for free" under MPS.

4. **But gen throughput still degrades 26.6% under MPS.** Despite the MoE architecture, the gen model still shows SM util at 100% during decode (dmon). This suggests vLLM's CUDA graph captures and continuous batching keep SMs busy even with only 3.3B active parameters per token. The MoE routing, attention layers, and vLLM scheduling overhead may fill SMs more than the raw arithmetic intensity would suggest.

5. **Memory bandwidth is NOT the bottleneck on H200.** Mem bandwidth utilization was only 18% during gen-only decode. On the 4090, it was 100%. The H200's 4.8 TB/s HBM3e bandwidth is so fast relative to the model's decode requirements that the GPU isn't bandwidth-bound at all — it's compute-bound even for MoE decode. This means the premise that "MoE decode leaves idle SMs because it's bandwidth-bound" does not hold on this hardware.

6. **The SM 100% utilization paradox.** Gen-only shows SM=100% at only 18% mem bandwidth and 230W power. This means the SMs are active but mostly waiting (stalled on memory fetches, or executing lightweight operations). They're "occupied" by vLLM's CUDA graphs but not doing useful compute. When embedding is added, the SMs have more actual work to do (power goes up to 400W), but since they were already "occupied," the embedding kernels must wait for free SM slots, causing the 26.6% gen degradation.

**Comparison to prior runs:**

| | Run 1 (4090) | Run 2 (4090) | Run 3 (H200) |
|---|---|---|---|
| Gen model | Qwen 2.5 3B (dense) | Llama 3.1 8B (dense) | Qwen3 30B-A3B (MoE, FP8) |
| Embed model | e5-large-v2 (335M) | e5-large-v2 (335M) | Qwen3-Embed-8B (8B) |
| Gen baseline | 869 tok/s | 61.4 tok/s | 195.2 tok/s |
| Gen under MPS | -59 to -73% | -60% | **-26.6%** |
| Gen under time-sliced | -45 to -58% | -54% | **-59.9%** |
| Embed under MPS | 420-433 seq/s | 364.7 seq/s | **335.8 seq/s** |
| Power throttling | Yes (severe) | Yes (severe) | **No** |
| Clock throttling | 2760→2325 MHz | 2745→2490 MHz | **1980→1980 MHz** |

**Conclusion:** The H200 eliminates the power throttling problem and shows that MPS is clearly superior to time-slicing for co-location (+33pp gen throughput preserved). However, the hypothesis that MoE decode leaves idle SMs for "free" embedding work is only partially validated — embedding throughput is nearly free (96% of standalone), but gen throughput still takes a 26.6% hit. The remaining gen degradation appears to come from SM scheduling contention (CUDA graphs occupy SMs even when compute is lightweight) rather than power or bandwidth limitations.

**Next steps:**
- ~~Try with `--enforce-eager` in vLLM to disable CUDA graphs and see if that changes SM occupancy patterns~~ Done — see nsys results below
- ~~Run nsys kernel tracing to verify kernel overlap and identify bottleneck kernels~~ Done — see nsys results below
- Test with higher gen concurrency (2, 4) to see how SM contention scales
- Investigate MPS thread percentage limits (e.g. `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=60`) to reserve SM capacity for the embedding model
- Re-run MPS co-location experiment with `--enforce-eager` to see if removing CUDA graphs reduces gen degradation

#### nsys kernel trace results (enforce-eager mode, single-process)

**350,506 kernels captured in 15s of steady-state MoE decode (concurrency=1, Qwen3-30B-A3B FP8).**

| Metric | Value |
|--------|-------|
| Total kernels | 350,506 |
| Mean kernel duration | 4.0us |
| Mean blocks/kernel | 49 (out of 132 SMs) |
| Theoretical SM occupancy (by kernel count) | **17.4%** |
| Duration-weighted SM occupancy | **43.7%** |
| Kernels using <50% SMs | 84.5% (by count), 57.2% (by time) |
| Kernels with just 1 block | 45.7% |
| Kernels saturating all 132 SMs | 11.6% |
| **GPU idle fraction** | **90.7%** |
| Total kernel compute time | 1,389ms / 14,994ms trace span |

**This is the smoking gun.** The GPU is idle 90.7% of the time during MoE decode at concurrency=1. Individual decode kernels are extremely short (mean 4us) with large gaps between them (mean 38.8us). Most kernels use only a fraction of the 132 SMs.

**Kernel category breakdown:**

| Category | Kernels | Time | Pct |
|----------|---------|------|-----|
| MoE (expert routing + gating) | 66,815 | 455.8ms | 32.8% |
| Other (memcpy, KV cache, etc.) | 121,104 | 364.6ms | 26.2% |
| GEMM (FP8 cutlass) | 40,089 | 206.4ms | 14.9% |
| Attention (flash-attn) | 28,493 | 182.9ms | 13.2% |
| Norm (RMSNorm) | 53,730 | 124.0ms | 8.9% |
| Activation (SiLU) | 40,275 | 55.9ms | 4.0% |

**Top kernels by time:**
1. `fused_moe_kernel` — 24.3% of time, mean 472 blocks (the only kernel that saturates SMs)
2. FP8 cutlass GEMM — 14.9%, mean 39 blocks (heavily underutilized)
3. Flash attention — 10.4%, mean 132 blocks (full SM utilization)
4. FP8 quantization — 6.8%, only 1 block each
5. Reduce kernel — 5.3%, only 4 blocks each

**Key insight: the SM=100% from dmon was entirely misleading.** CUDA graphs were keeping SMs "reserved" even though actual kernel execution only uses them 9.3% of the time. In enforce-eager mode (no CUDA graphs), the true picture emerges: decode is a rapid-fire sequence of tiny kernels with massive idle gaps. This is exactly the pattern where MPS can schedule embedding kernels into the gaps.

**Why the 26.6% gen degradation then?** Two possible explanations:
1. **CUDA graph overhead in the baseline**: The baseline run used CUDA graphs, which pre-capture kernel sequences and replay them with minimal CPU overhead. Adding MPS may interfere with graph replay timing.
2. **Memory bandwidth contention**: Even at 18% dmon mem util, the MoE kernels are bandwidth-sensitive. The embedding model's memory traffic (~14.4GB model weights per forward pass) competes for HBM bandwidth during the gaps between gen kernels.

**Implication for the experiment:** The co-location test (Step 2) used CUDA graphs, where kernels appear densely packed. Re-running with `--enforce-eager` might reduce the gen degradation since there would be genuine idle time for embedding kernels to execute without contention.

### Why this is likely 4090-specific

The power wall problem would be different on data center GPUs:

1. **Higher power headroom.** H100 SXM has 700W TDP. The gap between idle-decode power draw and TDP is much larger, so embedding compute is less likely to trigger throttling.
2. **HBM vs GDDR6X.** HBM3 on H100 delivers 3.35 TB/s (vs 1 TB/s on 4090) and is more power-efficient per GB/s. Decode needs proportionally fewer SMs to saturate bandwidth, leaving more genuinely idle SMs.
3. **Higher bandwidth-per-SM ratio.** H100 has 132 SMs with 3.35 TB/s — ~25 GB/s per SM. The 4090 has 128 SMs with 1 TB/s — ~8 GB/s per SM. Decode on H100 is more deeply bandwidth-bound and likely uses far fewer SMs per step.
4. **Separate power domains.** Some data center GPUs manage compute and memory power independently, so embedding compute wouldn't starve memory bandwidth the way it does on the 4090.

To validate the hypothesis properly, it would need to be tested on H100/A100 hardware where the power budget and bandwidth-per-SM ratios are fundamentally different.

### The baseline problem

The original plan compared co-located (both models on one GPU) vs isolated (one model per GPU). But co-location requires more VRAM than either model alone — so you need a bigger GPU or smaller batch sizes to fit both. This means the "baseline" and "experiment" run on different hardware configurations, making the comparison unfair.

### Setup: 2 GPUs with NVLink

Use two NVLink-connected GPUs for both configurations. This controls for hardware — same GPUs, same total HBM, same interconnect.

**Configuration A — Dedicated (one job per GPU):**
- GPU 0: Generative model (TP=1), full HBM available for KV cache
- GPU 1: Embedding model, full HBM available for batch activations
- No cross-GPU communication, no MPS

**Configuration B — Co-located (TP=2 + MPS):**
- Generative model sharded TP=2 across both GPUs (NVLink for all-reduce)
- Embedding model running on both GPUs (or one) via MPS, scavenging idle SMs during decode
- Generative model benefits from 2x HBM bandwidth via tensor parallelism
- Embedding work fills SMs that would otherwise idle during decode's memory-bound phase

### What this tests

The real capacity planning question: "Given N GPUs, should I dedicate some to embeddings and run the generative model at lower TP, or shard the generative model wider and co-locate embedding work via MPS?"

Configuration A maximizes simplicity and per-workload HBM. Configuration B trades NVLink communication overhead and shared HBM for higher SM utilization and 2x memory bandwidth for decode.

### What to measure

1. **Generative throughput (tokens/sec):** A should win on latency per token (no NVLink overhead), B might win on throughput if 2x bandwidth outweighs communication cost
2. **Embedding throughput (sequences/sec):** A has a full dedicated GPU. B shares GPU resources but might maintain comparable throughput if decode leaves enough idle SMs
3. **Aggregate throughput per GPU:** The key metric. Total useful work (tokens + embeddings) divided by number of GPUs. B wins if the TP=2 bandwidth gain + SM scavenging exceeds the NVLink overhead + memory contention costs
4. **GPU utilization breakdown:** SM occupancy, HBM bandwidth utilization, Tensor Core activity, and NVLink utilization across both configs (via `nsys` or `dcgmi`)
5. **Memory allocation:** HBM usage breakdown (weights, KV cache, activations) in both configs to understand headroom

### Expected outcome

B wins if:
- The generative model's decode phase is sufficiently memory-bandwidth-bound that TP=2 gives near-linear bandwidth scaling
- NVLink communication overhead for TP=2 is small relative to the bandwidth gain
- Enough SM headroom remains during decode to run meaningful embedding batches via MPS
- HBM can fit both models' weights plus KV cache plus embedding activations across two GPUs

B loses if:
- NVLink all-reduce overhead eats the bandwidth gain from TP=2
- Co-located embedding work interferes with generative decode enough to degrade its throughput
- Memory pressure forces significantly smaller generative batch sizes or embedding batch sizes

## Key risks

- **Prefill contention.** The generative model's prefill phase is also compute-bound. During prefill, both workloads compete for SMs and the complementary overlap disappears. High prefill-to-decode ratios reduce the benefit.
- **Memory pressure.** KV cache growth with concurrent generative sequences could squeeze out room for embedding batches, forcing smaller embedding batch sizes and reducing throughput.
- **MPS daemon reliability.** MPS is a single point of failure — a GPU fault in either process can take down both. Acceptable for batch processing, less so for always-on services.
- **Diminishing returns.** If the generative model's decode phase already achieves high SM occupancy (large batch sizes, GQA, etc.), there may be fewer idle SMs for embedding work to fill.

## Open questions

- **Why not just batch more?** The naive alternative to co-location is simply increasing the generative model's batch size to improve GPU utilization. Decode is bandwidth-bound, so larger batches don't help SM utilization — but does that hold at all batch sizes and model architectures? At what point does batching more become genuinely wasteful vs co-location becoming worthwhile? This needs empirical validation rather than assuming the answer.
