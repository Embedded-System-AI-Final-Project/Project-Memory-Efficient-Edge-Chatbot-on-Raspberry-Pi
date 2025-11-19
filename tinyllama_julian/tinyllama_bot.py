"""
TinyLlama chatbot for Raspberry Pi
- Uses llama.cpp via llama-cpp-python
- Enables sliding-window KV cache via n_ctx
- Enables quantized KV cache via kv_cache_type
- Logs latency, tokens/sec, memory, and CPU temperature to JSONL
"""

import json
import time
import psutil
import subprocess
from datetime import datetime

from llama_cpp import Llama

# ----------------------------
# Configuration
# ----------------------------

CONFIG = {
    "model_path": "tinyllama_julian/models/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf",
    "n_ctx": 512,
    "kv_cache_type": "q8_0",
    "n_threads": 4,
}


LOG_FILE = "tinyllama_julian/logs/tinyllama_logs.jsonl"


# ----------------------------
# Helpers
# ----------------------------

def get_cpu_temp_c():
    """Try to get CPU temperature on Raspberry Pi via vcgencmd. Returns float or None."""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        # Example output: "temp=51.0'C\n"
        return float(out.strip().split("=")[1].split("'")[0])
    except Exception:
        return None


def get_rss_mb():
    """Return current process RSS memory in MB."""
    proc = psutil.Process()
    rss_bytes = proc.memory_info().rss
    return rss_bytes / (1024 * 1024)


def log_metrics(meta: dict):
    """Append a JSON log line to LOG_FILE."""
    meta = dict(meta)  # make a copy we can mutate
    meta["timestamp"] = datetime.now().isoformat()

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

    # Also print a pretty version to the console
    print("\n[METRICS]")
    print(json.dumps(meta, indent=2))


# ----------------------------
# Model setup
# ----------------------------

print("Loading TinyLlama model… this may take a bit the first time.")

llm = Llama(
    model_path=CONFIG["model_path"],
    n_ctx=CONFIG["n_ctx"],
    kv_cache_type=CONFIG["kv_cache_type"],
    n_threads=CONFIG["n_threads"],
    use_mmap=True,   # map file from disk instead of fully loading to RAM
)

print("Model loaded. You can start chatting. Type 'exit' or 'quit' to stop.\n")


# ----------------------------
# Chat loop
# ----------------------------

SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant running on a small Raspberry Pi. "
    "Answer briefly and clearly.\n\n"
)

conversation_history = []  # list of {"role": "user"/"assistant", "content": str}


def build_prompt(user_input: str) -> str:
    """Builds a simple chat-style prompt from history + new user input."""
    lines = [SYSTEM_PROMPT]
    for msg in conversation_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    lines.append(f"User: {user_input}")
    lines.append("Assistant:")
    return "\n".join(lines)


def chat_once(user_input: str):
    """Run one chat turn and log metrics."""
    global conversation_history

    prompt = build_prompt(user_input)

    temp_before = get_cpu_temp_c()
    mem_before = get_rss_mb()
    t0 = time.time()

    # Call llama.cpp
    result = llm(
        prompt,
        max_tokens=128,
        temperature=0.7,
        stop=["User:", "Assistant:"],  # helps keep responses short
    )

    t1 = time.time()
    temp_after = get_cpu_temp_c()
    mem_after = get_rss_mb()

    # Extract model text and usage info
    text = result["choices"][0]["text"]
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", None)
    completion_tokens = usage.get("completion_tokens", None)
    total_tokens = usage.get("total_tokens", None)

    latency = t1 - t0
    tokens_generated = completion_tokens or 0
    tokens_per_s = tokens_generated / latency if latency > 0 else 0.0

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": text})

    # Prepare metrics record
    metrics = {
        "model_path": CONFIG["model_path"],
        "mode": "tinyllama_llamaccp_sliding_quant_kv",
        "n_ctx": CONFIG["n_ctx"],
        "kv_cache_type": CONFIG["kv_cache_type"],
        "n_threads": CONFIG["n_threads"],
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

    log_metrics(metrics)

    return text


def main():
    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user.strip().lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        reply = chat_once(user)
        print("\nAssistant:", reply, "\n")


if __name__ == "__main__":
    main()
