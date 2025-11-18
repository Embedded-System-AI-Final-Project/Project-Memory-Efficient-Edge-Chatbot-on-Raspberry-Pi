import sys
from llama_cpp import Llama

# Redirect output of LLM loading to a log, save original to restore
original_stderr = sys.stderr

logfile = open("llama_load.log", "w")
sys.stderr = logfile

llm = Llama.from_pretrained(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    n_ctx=2048,  # Sliding window (number of cache values to keep)
    kv_cache_type="q8_0",  # INT8 KV cache
    flash_attn=True,  # optional if your build supports it
)
sys.stderr = original_stderr
logfile.close()

for token in llm(
    "Hello ",
    max_tokens=50,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repeat_penalty=1.1,
    stream=True,
):
    print(token["choices"][0]["text"], end="", flush=True)
