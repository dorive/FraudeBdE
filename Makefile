# --------------------------------------------------------------------
# COMANDOS DISPONIBLES:
#   make up            -> Levanta Postgres y MLflow en segundo plano
#   make down          -> Apaga y limpia los contenedores
#   make ingerir_datos -> Ingiera los datos en la base de datos
#   make crear_features-> Crea las features y las guarda en la base de datos
#   make test          -> Corre los tests
# --------------------------------------------------------------------

.PHONY: up down train eval

up:
	docker compose up -d postgres mlflow

down:
	docker compose down -v

ingerir_datos:
	python -m pipelines.01_postgresql_init

crear_features:
	python -m pipelines.02_postgresql_features

test:
	pytest -q

