from collections import deque
"""
Class to hold conversational memory. Used so the bot can 
understand the conversation better
"""
class BotMemory:
    def __init__(self, max_mem = 10):
        self.memory = deque([])
        self.max_mem = max_mem

    def add_to_mem(self, role, text):
        "Add to memory, remove if we are larger than max memory size"
        self.memory.append({"role": role, "text": text})
        if len(self.memory) > self.max_mem:
            self.memory.popleft()

    def get_mem(self):
        return self.memory