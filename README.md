## TinyLlama (Julian Branch)

This branch implements a TinyLlama chatbot using `llama.cpp` with sliding-window KV-cache (`n_ctx`) and quantized KV-cache (`kv_cache_type`). Logs include latency, tokens/sec, memory usage, and CPU temperature.

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate    # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
