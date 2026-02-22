APP_MODULE=main:app
HOST=0.0.0.0
PORT=8000

.PHONY: dev start

dev:
	uv run uvicorn $(APP_MODULE) --reload --host $(HOST) --port $(PORT)

start:
	uv run uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT)
