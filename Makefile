.PHONY: tinyllama-bot setup fix clean

# -------------------------
# Run TinyLlama Bot
# -------------------------
tinyllama-bot:
	python -m tinyllama_julian.tinyllama_bot

# -------------------------
# Setup virtual environment
# Works on Linux + Raspberry Pi
# -------------------------
setup:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
	else \
		echo "Using existing virtual environment."; \
	fi
	@echo "Installing dependencies..."
	@venv/bin/pip install -r requirements.txt

# -------------------------
# Format code (optional)
# -------------------------
fix:
	black .

# -------------------------
# Clean environment
# -------------------------
clean:
	rm -rf venv
