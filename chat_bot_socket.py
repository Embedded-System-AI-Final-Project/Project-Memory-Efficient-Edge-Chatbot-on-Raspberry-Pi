import socket
import json
from llama_cpp import Llama
from bot_mem import BotMemory


HOST = "0.0.0.0"
PORT = 5001


def load_model():
    return Llama.from_pretrained(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        n_ctx=2048,
        kv_cache_type="q8_0",
        flash_attn=True,
        seed=-1,
    )


def format_prompt(memory, user_input):
    bot_init = "You are a helpful chatbot. Only respond as the Bot. Do not repeat or imitate the User."
    return f"{bot_init}\nConversation so far:\n{memory}\nUser: {user_input}\nBot:"


def generate_response(llm, memory, user_input):
    prompt = format_prompt(memory.get_mem(), user_input)
    memory.add_to_mem("user", user_input)

    response_text = ""

    for token in llm(
        prompt,
        max_tokens=256,
        temperature=0.9,
        top_p=0.95,
        top_k=100,
        repeat_penalty=1.1,
        stream=True,
        stop=["\nUser:", "User:", "Bot:", "\nConversation"],
    ):
        response_text += token["choices"][0]["text"]

    memory.add_to_mem("bot", response_text)
    return response_text


def main():
    print("Loading model...")
    llm = load_model()
    memory = BotMemory(5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen()
        print(f"LLM socket server active on port {PORT}, waiting for messages.")

        while True:
            conn, addr = server.accept()
            with conn:
                data = conn.recv(4096).decode("utf-8")

                if not data:
                    continue

                try:
                    msg = json.loads(data)
                    user_input = msg.get("text")
                except:
                    conn.sendall(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
                    continue

                response = generate_response(llm, memory, user_input)

                reply = {"response": response}
                conn.sendall(json.dumps(reply).encode("utf-8"))


if __name__ == "__main__":
    main()
