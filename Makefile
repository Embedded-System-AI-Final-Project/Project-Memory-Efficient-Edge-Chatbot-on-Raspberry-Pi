.PHONY: setup clean

setup:
	if [ "$$VIRTUAL_ENV" != "" ]; then \
		echo "Currently in a virtual environment: $$VIRTUAL_ENV"; \
	else \
		python -m venv venv; \
	fi
		source venv/bin/activate && pip install -r requirements.txt

fix:
	black .

clean:
	rm -rf venv
