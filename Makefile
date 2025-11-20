.PHONY: tinyllama-bot tinyllama-bench setup fix clean

tinyllama-bot:
	python -m tinyllama_julian.tinyllama_bot

tinyllama-bench:
	python -m tinyllama_julian.tinyllama_benchmark

setup:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
	else \
		echo "Using existing virtual environment."; \
	fi
	@echo "Installing dependencies..."
	@venv/bin/pip install -r requirements.txt

fix:
	black .

clean:
	rm -rf venv
