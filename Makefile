.PHONY: setup clean fix

setup:
	@if [ "$$VIRTUAL_ENV" != "" ]; then \
		echo "Currently in a virtual environment: $$VIRTUAL_ENV"; \
	else \
		python3 -m venv venv; \
	fi
	@. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

fix:
	black .

clean:
	rm -rf venv
