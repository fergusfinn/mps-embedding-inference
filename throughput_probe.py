"""
High-load offline throughput probe.
Matches the A/B test server conditions:
- C=512 concurrent sequences
- max_tokens=512
- Sustained generation (multiple rounds)

Reports throughput in tok/s, directly comparable to gen_benchmark.py results.
"""
import torch
import time
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-30B-A3B-FP8",
    gpu_memory_utilization=0.50,
    max_model_len=4096,
    max_num_seqs=512,
    enforce_eager=True,
    disable_log_stats=True,
)

prompts = ["Write a detailed essay about the history of computing. Start from the very beginning and cover every major development in great detail."] * 512
sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

# Warmup
print("Warmup...")
warmup_prompts = prompts[:64]
warmup_params = SamplingParams(temperature=0.0, max_tokens=64)
llm.generate(warmup_prompts, warmup_params)

# Measure
print("Measuring throughput (512 prompts x 512 tokens)...")
t0 = time.monotonic()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.monotonic() - t0

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / elapsed
print(f"Generated {total_tokens} tokens in {elapsed:.1f}s")
print(f"Throughput: {throughput:.1f} tokens/sec")
