.PHONY: setup clean

setup:
	if [ "$$VIRTUAL_ENV" != "" ]; then \
		echo "Currently in a virtual environment: $$VIRTUAL_ENV"; \
	else \
		python -m venv venv; \
	fi
	. venv/bin/activate && pip install -r requirements.txt

clean:
	rm -rf venv
