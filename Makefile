.PHONY: install test lint demo docker
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check src tests && mypy src
demo:
	rag-harness bench
docker:
	docker build -t agentic-rag-evaluation-harness . && docker run --rm agentic-rag-evaluation-harness
