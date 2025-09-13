CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS scoring;

-- ejemplo de tabla cruda (ajústala en 01_ingest.py)
CREATE TABLE IF NOT EXISTS raw.transactions (
  id SERIAL PRIMARY KEY,
  tx_time TIMESTAMP NOT NULL,
  amount DOUBLE PRECISION,
  class SMALLINT,
  -- añade columnas según dataset seleccionado
  payload JSONB
);
