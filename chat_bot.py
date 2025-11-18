import os, time, json, psutil, threading

import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

from bot_mem import BotMemory

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_cpu_temp():
    try:
        import subprocess

        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        return float(out.strip().split("=")[1].split("'")[0])
    except Exception:
        return None


def get_mem_over_time(interval=0.1):
    """
    Function that will return two functions to be called later
    These functions serve to check memory ever interval to show
    change over time
    - start(): starts memory sampling in a background thread
    - stop(): stops sampling and returns collected data
    """

    mem_log = []
    running = False
    process = psutil.Process(os.getpid())
    thread = None
    start_time = 0

    def sampler():
        while running:
            t = time.time() - start_time
            mem_log.append((t, (process.memory_info().rss) / (1024 * 1024)))
            time.sleep(interval)

    def start():
        nonlocal running
        running = True
        nonlocal start_time
        if start_time != 0:
            start_time = time.time()
        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        return thread

    def stop():
        nonlocal running
        running = False
        time.sleep(interval * 2)  # allow final sample
        if thread is not None:
            thread.join()
        return mem_log

    return start, stop


def prompt_gen(mem, user_input):
    bot_init = "You are a concise, helpful chatbot. Only provide one response."
    return f"{bot_init}\nConversation so far:\n{mem}\nUser: {user_input}\nBot:"


def main():
    model_name = "distilgpt2"
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    new_tokens = 20

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    start_mem, stop_mem = get_mem_over_time(interval=0.1)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Need a way to exit the chat
    break_out = False
    prompt = None
    bot_mem = BotMemory(5)
    while True:
        while prompt == None:
            prompt = input("Ask the model something: ").strip().lower() or None
            if prompt == "exit":
                print("Thank you for using this bot")
                break_out = True
                break
            if prompt == None:
                print("Empty input not allowed")
        if break_out:
            break

        t_start = datetime.now().isoformat()
        mem_before = psutil.Process(os.getpid()).memory_info().rss
        cpu_t_before = get_cpu_temp()
        t0 = time.time()

        bot_prompt = prompt_gen(bot_mem.get_mem(), prompt)
        bot_mem.add_to_mem("user", prompt)
        inputs = tokenizer(bot_prompt, return_tensors="pt")
        prompt = None
        start_mem()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                use_cache=True,
            )
        mem_samples = stop_mem()
        t1 = time.time()

        mem_after = psutil.Process(os.getpid()).memory_info().rss
        cpu_t_after = get_cpu_temp()

        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        bot_mem.add_to_mem("bot", decoded)
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
            "cpu_temp_c_end": cpu_t_after,
        }

        with open("baseline_log.json", "a") as f:
            f.write(json.dumps(log) + "\n")
        print("\n=== METRICS ===\n")
        print(json.dumps(log, indent=2))

        print(decoded)


if __name__ == "__main__":
    main()
