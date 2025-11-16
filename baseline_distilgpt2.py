import os, time, json, psutil
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_cpu_temp():
    try: 
        import subprocess
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text = True)
        return float(out.strip().split('=')[1].split("'")[0])
    except Exception:
        return None
    
def main():
    model_name = "distilgpt2"
    prompt = "How does one get in contact with the start-up company Rogue C&E?\n"
    new_tokens = 64
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    t_start = datetime.now().isoformat()
    mem_before = psutil.Process(os.getpid()).memory_info().rss
    cpu_t_before = get_cpu_temp()
    
    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors = "pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens = new_tokens,
        do_sample = False,
        temperature = 0.8,
        top_p = 0.95,
        repetition_penalty = 1.1
    )
    t1 = time.time()
    
    mem_after = psutil.Process(os.getpid()).memory_info().rss
    cpu_t_after = get_cpu_temp()
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens = True)
    prompt_len = inputs["input_ids"].shape[1]
    total_len = int(outputs.shape[1])
    new_tokens_actual = max(total_len - prompt_len, 0)

    elapsed = max(t1 - t0, 1e-9)
    tokens_per_s = new_tokens_actual / elapsed
    max_rss_mb = (mem_after - mem_before) / (1024 * 1024)

    log = {
        "timestamp_start": t_start,
        "timestamp_end": datetime.now().isoformat(),
        "model": model_name,
        "prompt_len": prompt_len,
        "new_tokens_requested": new_tokens,
        "new_tokens_actual": new_tokens_actual,
        "latency": round(t1 - t0, 2),
        "tokens_per_s": round(tokens_per_s, 2),
        "max_rss_mb": round(max_rss_mb, 2),
        "cpu_temp_c_start": cpu_t_before,
        "cpu_temp_c_end": cpu_t_after
    }
    
    print("\n=== SAMPLE OUTPUT ===\n")
    print(decoded[:500] + "...")
    with open("baseline_log.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    print("\n=== METRICS ===\n")
    print(json.dumps(log, indent=2))

if __name__ == "__main__":
    main()