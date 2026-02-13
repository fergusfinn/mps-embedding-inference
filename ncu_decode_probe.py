"""
Minimal decode probe for ncu profiling.
Loads the gen model, fills KV cache with prefills, then does decode steps.
The decode steps are what we want to profile with ncu.
"""
import torch
import time
from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="Qwen/Qwen3-30B-A3B-FP8",
    gpu_memory_utilization=0.50,
    max_model_len=4096,
    max_num_seqs=512,
    enforce_eager=True,  # disable cuda graphs so ncu can see individual kernels
)

# Generate with multiple concurrent sequences to simulate decode load
prompts = ["Write a detailed essay about the history of computing."] * 64
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

print("Starting generation (this is what ncu profiles)...")
torch.cuda.nvtx.range_push("decode_phase")
outputs = llm.generate(prompts, sampling_params)
torch.cuda.nvtx.range_pop()

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"Generated {total_tokens} tokens from {len(prompts)} prompts")
