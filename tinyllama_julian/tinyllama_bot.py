from llama_cpp import Llama
import time

# Load model with sliding window + quantized KV-cache
llm = Llama(
    model_path="tinyllama.gguf",  
    n_ctx=512,               # sliding window
    kv_cache_type="q8_0",    # quantized KV cache
    n_threads=4,
    use_mmap=True,
)

while True:
    user = input("You: ")

    if user.lower() in ["exit", "quit"]:
        break

    t0 = time.time()
    output = llm(user, max_tokens=128)
    t1 = time.time()

    reply = output["choices"][0]["text"]
    print(f"Bot ({t1 - t0:.2f}s):", reply)
