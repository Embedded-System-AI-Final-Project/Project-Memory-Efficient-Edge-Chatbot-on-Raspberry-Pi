import sys, time
import os, psutil, threading
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
    bot_init = "You are a helpful chatbot. Only respond as the Bot. Do not repeat or imitate the User."
    return f"{bot_init}\nConversation so far:\n{mem}\nUser: {user_input}\nBot:"


def main():

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

    chat_mem = BotMemory(5)

    start_mem, stop_mem = get_mem_over_time(interval=0.1)

    # Need a way to exit the chat
    break_out = False
    prompt = None

    logfile_2 = open("llama_run_output.log", "a")

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
        # Redirect stderr to log_file
        sys.stderr = logfile_2
        for token in llm(
            bot_prompt,
            max_tokens=50,
            temperature=0.9,
            top_p=0.95,
            top_k=100,
            repeat_penalty=1.1,
            stream=True,
            # Strings to stop the bot from responding as a user
            stop=["\nUser:", "User:", "Bot:", "user:", "bot:", "\nConversation"],
        ):
            text = token["choices"][0]["text"]
            print(text, end="", flush=True)
            bot_response += text
        mem_log = stop_mem()
        print("\n")
        # Redirect error back to proper output
        sys.stderr = original_stderr
        chat_mem.add_to_mem("bot", bot_response)
    logfile_2.close()


if __name__ == "__main__":
    main()
