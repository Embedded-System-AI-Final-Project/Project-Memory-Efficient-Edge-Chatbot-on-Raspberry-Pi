import sys, time, json
import os, psutil, threading
from datetime import datetime
from llama_cpp import Llama
from bot_mem import BotMemory


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
    start_time = 0.0

    def sampler():
        while running:
            t = time.time() - start_time
            mem_log.append((t, (process.memory_info().rss) / (1024 * 1024)))
            time.sleep(interval)

    def start():
        nonlocal running, start_time, thread
        running = True
        start_time = time.time()
        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        return thread

    def stop():
        nonlocal running, thread
        running = False
        time.sleep(interval * 2)  # allow final sample
        if thread is not None:
            thread.join()
        return mem_log

    return start, stop


def prompt_gen(mem, user_input):
    bot_init = "You are a helpful chatbot. Only respond as the Bot. Do not repeat or imitate the User."
    return f"{bot_init}\nConversation so far:\n{mem}\nUser: {user_input}\nBot:"


def main():

    # Redirect output of LLM loading to a log, save original to restore
    original_stderr = sys.stderr
    logfile = open("llama_load.log", "w")
    sys.stderr = logfile
    
    t_start = datetime.now().isoformat()
    
    llm = Llama.from_pretrained(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        n_ctx=2048,  # Sliding window (number of cache values to keep)
        kv_cache_type="q8_0",  # INT8 KV cache
        flash_attn=True,
        seed=-1, # Random seed
    )
    
    sys.stderr = original_stderr
    logfile.close()

    chat_mem = BotMemory(5)

    start_mem, stop_mem = get_mem_over_time(interval=0.1)

    # Need a way to exit the chat
    break_out = False
    prompt = None

    logfile_2 = open("llama_run_output.log", "a")
    metrics_log = open("llama_metrics.jsonl", "a", encoding="utf-8")

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

        # Generate prompt from input
        bot_prompt = prompt_gen(chat_mem.get_mem(), prompt)
        chat_mem.add_to_mem("user", prompt)
        prompt = None
        start_mem()

        bot_response = ""
        token_count = 0
        
        # Measure Latency
        t0 = time.time()
        
        # Redirect stderr to log_file
        sys.stderr = logfile_2
        for token in llm(
            bot_prompt,
            max_tokens=256,
            temperature=0.9,
            top_p=0.95,
            top_k=100,
            repeat_penalty=1.1,
            stream=True,
            # Strings to stop the bot from responding as a user
            stop=["\nUser:", "User:", "Bot:", "user:", "bot:", "\nConversation", "conversation"],
        ):
            text = token["choices"][0]["text"]
            print(text, end="", flush=True)
            bot_response += text
            token_count += 1
            
        t1 = time.time()
        mem_log = stop_mem()
        print("\n")
        # Redirect error back to proper output
        sys.stderr = original_stderr
        chat_mem.add_to_mem("bot", bot_response)
        
        #======METRICS======
        latency_s = t1 - t0
        token_per_s = token_count / latency_s if latency_s > 0 else 0.0
        max_rss_mb  = max((mb for (_, mb) in mem_log), default = None) 
        
        metrics_record = {
            "timestamp": t_start,
            "prompt": bot_prompt,
            "response": bot_response,
            "latency_s": round(latency_s, 3),
            "tokens_generated": token_count,
            "token_per_s": round(token_per_s, 3),
            "max_rss_mb": round(max_rss_mb, 2) if max_rss_mb is not None else None
        }
        metrics_log.write(json.dumps(metrics_record) + "\n")
        metrics_log.flush()
        
    logfile_2.close()
    metrics_log.close()


if __name__ == "__main__":
    main()
