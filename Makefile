.PHONY: up down train eval api ui lint test demo

up:
	docker compose up -d postgres mlflow

down:
	docker compose down -v

api:
	docker compose up -d api

ui:
	docker compose up -d streamlit

lint:
	ruff check .

test:
	pytest -q

train:
	python pipelines/03_train.py

eval:
	python pipelines/04_eval.py

demo:
	make up && sleep 3 && make api && make ui
