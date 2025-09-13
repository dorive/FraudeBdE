# --------------------------------------------------------------------
# COMANDOS DISPONIBLES:
#   make up            -> Levanta Postgres y MLflow en segundo plano
#   make down          -> Apaga y limpia los contenedores
#   make build_api     -> Compila la imagen Docker de la API
#   make api           -> Levanta el servicio de la API
#   make ui            -> Levanta la interfaz de usuario (Streamlit)
#   make lint          -> Ejecuta el linter (Ruff)
#   make test          -> Ejecuta los tests (pytest)
#   make train         -> Entrena el modelo
#   make eval          -> Evalúa el modelo
#   make demo          -> Levanta todo (Postgres, MLflow, API, UI)
#   make features      -> Construye las features y las guarda en ./data
#   make features_pg   -> Construye las features y las guarda en Postgres
#   make score_batch   -> Ejecuta el batch scoring (local o API)
# --------------------------------------------------------------------

.PHONY: up down train eval api ui lint test demo

up:
	docker compose up -d postgres mlflow

down:
	docker compose down -v

build_api:
	docker compose build api

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

features:
	python -m pipelines.02_feature_build

features_pg:
	python -m pipelines.02b_features_to_postgres

score_batch:
	python -m pipelines.05_batch_scoring
