"""
Generative model benchmark client.
Sends concurrent requests to a vLLM server and measures decode throughput.

Usage:
    python gen_benchmark.py --base-url http://localhost:8100 --duration 60
"""
import argparse
import asyncio
import time
import aiohttp
import json


async def generate_request(session, base_url, prompt, max_tokens, model):
    """Send a single chat completion request and return token count + duration."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    t0 = time.monotonic()
    async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
        data = await resp.json()
        elapsed = time.monotonic() - t0
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        return completion_tokens, elapsed


async def run_benchmark(base_url, duration, concurrency, max_tokens, model):
    """Run continuous requests for `duration` seconds with `concurrency` workers."""
    prompt = "Write a detailed essay about the history of computing. Start from the very beginning and cover every major development in great detail."

    total_tokens = 0
    total_requests = 0
    request_stats = []  # (tokens, elapsed) per request
    start = time.monotonic()

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
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
                request_stats.append((tokens, elapsed))

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    wall_time = time.monotonic() - start
    print(f"\n=== Generative Benchmark Results ===")
    print(f"Duration: {wall_time:.1f}s")
    print(f"Total requests: {total_requests}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Throughput: {total_tokens / wall_time:.1f} tokens/sec")
    print(f"Avg tokens/request: {total_tokens / max(total_requests, 1):.1f}")

    if request_stats:
        latencies = sorted([e for _, e in request_stats])
        per_token = sorted([e / t for t, e in request_stats if t > 0])
        n = len(latencies)
        print(f"\n--- Per-request latency ---")
        print(f"  p50: {latencies[n//2]:.2f}s  p95: {latencies[int(n*0.95)]:.2f}s  p99: {latencies[int(n*0.99)]:.2f}s")
        if per_token:
            m = len(per_token)
            print(f"--- Per-token latency ---")
            print(f"  p50: {per_token[m//2]*1000:.1f}ms  p95: {per_token[int(m*0.95)]*1000:.1f}ms  p99: {per_token[int(m*0.99)]*1000:.1f}ms")
            print(f"  Decode rate (p50): {1/per_token[m//2]:.1f} tok/s/request")

    return total_tokens / wall_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8100")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.base_url, args.duration, args.concurrency, args.max_tokens, args.model))
