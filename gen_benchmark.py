"""
Generative model benchmark client.
Sends concurrent requests to a vLLM server and measures decode throughput.

Usage:
    python gen_benchmark.py --base-url http://localhost:8000 --duration 60
"""
import argparse
import asyncio
import time
import aiohttp
import json


async def generate_request(session, base_url, prompt, max_tokens, model):
    """Send a single completion request and return token count + duration."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.monotonic()
    async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
        data = await resp.json()
        elapsed = time.monotonic() - t0
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        return completion_tokens, elapsed


async def run_benchmark(base_url, duration, concurrency, max_tokens, model):
    """Run continuous requests for `duration` seconds with `concurrency` workers."""
    # Short prompt to minimize prefill, maximize decode
    prompt = "Write a detailed essay about the history of computing. Start from the very beginning and cover every major development in great detail."

    total_tokens = 0
    total_requests = 0
    start = time.monotonic()

    async with aiohttp.ClientSession() as session:
        # Warm up with a single request
        print("Warming up...")
        await generate_request(session, base_url, prompt, 10, model)

        print(f"Benchmarking for {duration}s with {concurrency} concurrent requests, max_tokens={max_tokens}...")
        start = time.monotonic()

        async def worker():
            nonlocal total_tokens, total_requests
            while time.monotonic() - start < duration:
                tokens, elapsed = await generate_request(session, base_url, prompt, max_tokens, model)
                total_tokens += tokens
                total_requests += 1

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    wall_time = time.monotonic() - start
    print(f"\n=== Generative Benchmark Results ===")
    print(f"Duration: {wall_time:.1f}s")
    print(f"Total requests: {total_requests}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Throughput: {total_tokens / wall_time:.1f} tokens/sec")
    print(f"Avg tokens/request: {total_tokens / max(total_requests, 1):.1f}")
    return total_tokens / wall_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.base_url, args.duration, args.concurrency, args.max_tokens, args.model))
