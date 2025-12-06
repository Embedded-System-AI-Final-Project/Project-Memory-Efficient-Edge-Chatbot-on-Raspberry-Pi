# paged_mem.py
#   - Memory consumption vs sequence length
#   - Latency vs sequence length
#
# Run:
#     python paged_mem.py
#
# This generates:
#   - paged_raw_results.csv
#   - paged_summary.csv
#   - paged_memory_vs_seq.png
#   - paged_latency_vs_seq.png

import time
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1.  PagedMemory 
# ---------------------------------------------------------------------------

class PagedMemory:
    def __init__(self, max_pages=4, page_token_limit=256):
        """
        max_pages: how many pages we keep in memory
        page_token_limit: max approximate tokens (we use word count) per page
        """
        self.max_pages = max_pages
        self.page_token_limit = page_token_limit
        self.pages = [[]]
        self.current_tokens = 0

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def add_to_mem(self, role: str, text: str):
        """
        Add a new message (user or bot) into the paged cache.
        """
        tokens = self._count_tokens(text)

        # Check page overflow
        if self.current_tokens + tokens > self.page_token_limit:
            # start a new page
            self.pages.append([])
            self.current_tokens = 0

            # rotate (drop oldest) if max_pages exceeded
            if len(self.pages) > self.max_pages:
                self.pages.pop(0)

        self.pages[-1].append((role, text))
        self.current_tokens += tokens

    def get_mem(self) -> str:
        lines = []
        for page in self.pages:
            for role, text in page:
                prefix = "User" if role.lower() == "user" else "Bot"
                lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    # helper for benchmarking
    def approx_cache_bytes(self) -> int:
        total_chars = 0
        for page in self.pages:
            for _, text in page:
                total_chars += len(text)
        return total_chars


# ---------------------------------------------------------------------------
# 2. Benchmarking for PagedMemory
# ---------------------------------------------------------------------------

def benchmark_paged_memory(
    seq_lengths=None,
    n_repeats=3,
    max_pages=4,
    page_token_limit=256
):
    """
    Measures:
      - latency per token
      - approx memory usage in bytes
    for your PagedMemory cache at different sequence lengths.
    """
    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024, 1536, 2048]

    results = []

    for L in seq_lengths:
        for rep in range(n_repeats):
            cache = PagedMemory(max_pages=max_pages,
                                page_token_limit=page_token_limit)

            start = time.perf_counter()
            for i in range(L):
                cache.add_to_mem("user", f"tok_{i}")
            end = time.perf_counter()

            elapsed = end - start
            latency_per_token = (elapsed / L) * 1000.0  # ms
            cache_bytes = cache.approx_cache_bytes()
            cache_MB = cache_bytes / (1024 * 1024)

            results.append({
                "seq_length": L,
                "run": rep,
                "latency_ms": latency_per_token,
                "cache_bytes": cache_bytes,
                "cache_MB": cache_MB
            })

    return results


def run_benchmark():
    print("\nRunning PagedMemory benchmark...")

    data = benchmark_paged_memory()
    df = pd.DataFrame(data)
    df.to_csv("paged_raw_results.csv", index=False)

    # Summary table
    summary = (
        df.groupby("seq_length")
        .agg(
            avg_latency_ms=("latency_ms", "mean"),
            std_latency_ms=("latency_ms", "std"),
            avg_cache_MB=("cache_MB", "mean"),
            std_cache_MB=("cache_MB", "std"),
        )
        .reset_index()
    )
    summary.to_csv("paged_summary.csv", index=False)

    print("\n=== PagedMemory Performance Summary ===")
    print(summary.to_markdown(index=False, floatfmt=".6f"))

    # -------------------------------------------------------------
    # 3. Plots
    # -------------------------------------------------------------

    # Memory vs sequence length
    plt.figure()
    plt.plot(
        summary["seq_length"], summary["avg_cache_MB"],
        marker="o", linewidth=2
    )
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Approx Memory (MB)")
    plt.title("PagedMemory: Memory Consumption vs Sequence Length")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("paged_memory_vs_seq.png", dpi=200)

    # Latency vs sequence length
    plt.figure()
    plt.plot(
        summary["seq_length"], summary["avg_latency_ms"],
        marker="s", linewidth=2, color="orange"
    )
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Latency per Token (ms)")
    plt.title("PagedMemory: Latency vs Sequence Length")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("paged_latency_vs_seq.png", dpi=200)

    print(
        "\nBenchmark outputs saved:\n"
        "  - paged_raw_results.csv\n"
        "  - paged_summary.csv\n"
        "  - paged_memory_vs_seq.png\n"
        "  - paged_latency_vs_seq.png\n"
    )


if __name__ == "__main__":
    run_benchmark()
