# paged_mem.py
# Simple paged / rotating conversation cache for the chatbot.

class PagedMemory:
    def __init__(self, max_pages=4, page_token_limit=256):
        """
        max_pages: how many pages we keep in memory
        page_token_limit: max approximate tokens (we use word count) per page
        """
        self.max_pages = max_pages
        self.page_token_limit = page_token_limit
        # each page is a list of (role, text) tuples
        self.pages = [[]]
        self.current_tokens = 0

    def _count_tokens(self, text: str) -> int:
        # very rough token approximation: split by whitespace
        return len(text.split())

    def add_to_mem(self, role: str, text: str):
        """
        Add a new message (user or bot) into the paged cache.
        When the current page is full, start a new page and,
        if needed, rotate (drop) the oldest page.
        """
        tokens = self._count_tokens(text)

        # If adding this would overflow the current page, move to a new one
        if self.current_tokens + tokens > self.page_token_limit:
            # start a new page
            self.pages.append([])
            self.current_tokens = 0

            # If we now have too many pages, rotate (drop oldest)
            if len(self.pages) > self.max_pages:
                self.pages.pop(0)

        # Add message to the current (last) page
        self.pages[-1].append((role, text))
        self.current_tokens += tokens

    def get_mem(self) -> str:
        """
        Return the conversation history in the current pages
        as a single formatted string.
        """
        lines = []
        for page in self.pages:
            for role, text in page:
                # role is expected "user" or "bot"
                prefix = "User" if role.lower() == "user" else "Bot"
                lines.append(f"{prefix}: {text}")
        return "\n".join(lines)
