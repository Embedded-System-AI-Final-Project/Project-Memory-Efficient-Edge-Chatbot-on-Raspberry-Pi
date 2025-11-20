"""
TinyLlama benchmarking script.

Runs TinyLlama with different (n_ctx, kv_cache_type) configurations and
measures:
- latency (s)
- tokens/sec
- RSS memory (MB)
- CPU temperature (C)

Results are written to:
  tinyllama_julian/logs/tinyllama_benchmarks.csv
"""

import csv
import json
import time
import psutil
import subprocess
from datetime import datetime

from llama_cpp import Llama

MODEL_PATH = "tinyllama_julian/models/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"

# Config grid to test: edit these to change sliding window + kv-cache quantization
N_CTX_VALUES = [256, 512, 1024, 2048]           # sliding window sizes to test
KV_TYPES = ["f32", "q8_0", "f16", "q4_0"]          # kv_cache_type values to test
N_THREADS = 4                       # CPU threads
RUNS_PER_CONFIG = 3                 # how many times to repeat each config

# Prompt for benchmarking (same every run)
PROMPT = (
    "You are a helpful assistant running on a Raspberry Pi. "
    "Explain in a concise paragraph how sliding-window KV-cache "
    "and quantized KV-cache reduce memory usage for large language models."
)

MAX_TOKENS = 128

CSV_PATH = "tinyllama_julian/logs/tinyllama_benchmarks.csv"


def get_cpu_temp_c():
    """Try to get CPU temperature on Raspberry Pi via vcgencmd. Returns float or None."""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        return float(out.strip().split("=")[1].split("'")[0])
    except Exception:
        return None


def get_rss_mb():
    """Return current process RSS memory in MB."""
    proc = psutil.Process()
    rss_bytes = proc.memory_info().rss
    return rss_bytes / (1024 * 1024)


def run_single_benchmark(n_ctx: int, kv_type: str, run_idx: int) -> dict:
    """Run one benchmark for a given (n_ctx, kv_type)."""

    print(f"\n=== Config n_ctx={n_ctx}, kv_cache_type={kv_type}, run={run_idx+1} ===")

    # Load model for this config
    t_load0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=n_ctx,
        kv_cache_type=kv_type,
        n_threads=N_THREADS,
        use_mmap=True,
    )
    t_load1 = time.time()

    temp_before = get_cpu_temp_c()
    mem_before = get_rss_mb()
    t0 = time.time()

    # Make one generation call
    result = llm(
        PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        stop=["User:", "Assistant:"],
    )

    t1 = time.time()
    temp_after = get_cpu_temp_c()
    mem_after = get_rss_mb()

    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", None)
    completion_tokens = usage.get("completion_tokens", None)
    total_tokens = usage.get("total_tokens", None)

    latency = t1 - t0
    tokens_generated = completion_tokens or 0
    tokens_per_s = tokens_generated / latency if latency > 0 else 0.0

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "n_ctx": n_ctx,
        "kv_cache_type": kv_type,
        "run_idx": run_idx,
        "load_time_s": round(t_load1 - t_load0, 3),
        "latency_s": round(latency, 3),
        "tokens_generated": tokens_generated,
        "tokens_per_s": round(tokens_per_s, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "rss_mb_before": round(mem_before, 2),
        "rss_mb_after": round(mem_after, 2),
        "cpu_temp_c_before": temp_before,
        "cpu_temp_c_after": temp_after,
    }

    # Print a short preview of output text
    text = result["choices"][0]["text"]
    print("Sample output:", json.dumps(text[:120] + "..."))

    print("Metrics:", json.dumps(metrics, indent=2))
    return metrics


def main():
    all_rows = []

    for n_ctx in N_CTX_VALUES:
        for kv_type in KV_TYPES:
            for run_idx in range(RUNS_PER_CONFIG):
                row = run_single_benchmark(n_ctx, kv_type, run_idx)
                all_rows.append(row)

    # Write CSV
    fieldnames = list(all_rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nBenchmark complete. Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
