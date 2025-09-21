"""
Pipeline de preprocesamiento de los datos originales y guardado en PostgreSQL.
"""

import os, json
import pandas as pd
from typing import Iterable, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import Integer, Float, Text, String
from sqlalchemy.engine import Engine
from dotenv import load_dotenv
from ast import literal_eval


############################################################
# Carga de los datos originales
############################################################

df = pd.read_pickle("data/orig/FraudNLP_dataset.pkl")
action_vocab = pd.read_csv("data/orig/vocab.csv")

# Sanity check
broken_times = df[df.times.apply(lambda x: x[-1]!="]")]
assert broken_times.shape[0] == 1
assert broken_times.iloc[0].is_fraud==0

# Ignorar la fila corrupta
df = df[df.times.apply(lambda x: x[-1]=="]")]

# Preprocesamiento de las acciones
action_names = action_vocab.Name.to_list()
id_to_action = {str(i):a for i,a in enumerate(action_names)}
action_to_id = {a:str(i) for i,a in enumerate(action_names)}
df.actions = df.actions.apply(literal_eval)
df['n_actions'] = df['actions'].apply(len)
df.head(2)



############################################################
# Guardar en PostgreSQL
############################################################

load_dotenv()

# Host según si estamos en Docker o no
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")
PG_HOST = "postgres" if RUNNING_IN_DOCKER else "localhost"

# -----------------------------------------------------------------------------
# 1) Crear el engine de SQLAlchemy
# -----------------------------------------------------------------------------
# Construimos la URL leyendo credenciales del entorno (.env) y usando PG_HOST,
# que debería definirse previamente como "postgres" si ejecutas dentro de Docker.
engine: Engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{PG_HOST}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}",
    pool_pre_ping=True,  # hace un ping antes de usar la conexión -> evita conexiones zombi
)


SCHEMA = "fraudnlp"


# -----------------------------------------------------------------------------
# 2) Utilidades de limpieza de columnas
# -----------------------------------------------------------------------------
def sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de columnas a snake_case para evitar problemas en SQL.

    - Convierte a minúsculas.
    - Sustituye espacios, barras y guiones por guion bajo.
    - Elimina espacios al principio/fin.
    """
    def to_snake(s: str) -> str:
        return (
            s.strip()
             .lower()
             .replace(" ", "_")
             .replace("/", "_")
             .replace("-", "_")
        )

    df2 = df.copy()
    df2.columns = [to_snake(c) for c in df2.columns]
    return df2


# -----------------------------------------------------------------------------
# 3) Serialización segura de columnas "lista/dict" a JSON (para JSONB)
# -----------------------------------------------------------------------------
def prepare_json_cols(df: pd.DataFrame, cols_json: Iterable[str]) -> pd.DataFrame:
    """
    Serializa a JSON las columnas indicadas (listas/dicts) para poder insertarlas
    en PostgreSQL como JSONB.

    Maneja correctamente None y NaN escalares (no intenta json.dumps sobre ellos).
    """
    df2 = df.copy()
    for c in cols_json:
        if c in df2.columns:
            def safe_json(x):
                # None -> NULL en SQL
                if x is None:
                    return None
                # NaN escalar -> NULL en SQL
                if isinstance(x, float) and pd.isna(x):
                    return None
                # listas/dicts/cadenas -> JSON
                return json.dumps(x)
            df2[c] = df2[c].apply(safe_json)
    return df2


# -----------------------------------------------------------------------------
# 4) Mapeo de dtypes de pandas a tipos de PostgreSQL (incluye JSONB)
# -----------------------------------------------------------------------------
def dtype_map_jsonb(df: pd.DataFrame, json_cols: Iterable[str]) -> Dict[str, object]:
    """
    Devuelve un dict {columna: TipoSQLAlchemy} para tipar explícitamente las columnas
    al hacer to_sql. Esto ayuda a:
      - Guardar ciertas columnas como JSONB (acciones, tiempos…)
      - Evitar inferencias erróneas en textos largos
    """
    m: Dict[str, object] = {}
    json_cols = set(json_cols)

    for c in df.columns:
        if c in json_cols:
            m[c] = JSONB()
        elif pd.api.types.is_integer_dtype(df[c]):
            m[c] = Integer()
        elif pd.api.types.is_float_dtype(df[c]):
            m[c] = Float()
        elif pd.api.types.is_string_dtype(df[c]):
            # Si esperas textos largos, usa Text(); si no, un String(n) puede bastar
            maxlen = df[c].map(lambda x: len(str(x)) if pd.notna(x) else 0).max()
            m[c] = Text() if maxlen and maxlen > 255 else String(255)
        else:
            # Fallback conservador
            m[c] = Text()

    return m


# -----------------------------------------------------------------------------
# 5) Función principal: guardar DataFrame como tabla en PostgreSQL (JSONB)
# -----------------------------------------------------------------------------
def guardar_df_jsonb(df: pd.DataFrame, table_name: str, schema: str = SCHEMA) -> None:
    """
    Guarda un DataFrame en PostgreSQL como tabla, serializando columnas de tipo
    lista/dict a JSONB (útil para secuencias como actions/times).

    Pasos:
      1) Normaliza nombres de columnas a snake_case (sanitize_cols).
      2) Serializa a JSON las columnas secuenciales ('actions', 'times').
      3) Asegura que el esquema existe.
      4) Inserta por lotes con to_sql(method="multi", chunksize=10_000).

    Parámetros
    ---------
    df : pd.DataFrame
        Datos a guardar.
    table_name : str
        Nombre de la tabla destino (sin esquema).
    schema : str
        Esquema de PostgreSQL (por defecto, SCHEMA).

    Efectos
    -------
    Crea/reemplaza la tabla schema.table_name y escribe todas las filas de df.
    """
    # 1) Limpia columnas
    df_norm = sanitize_cols(df)

    # 2) Columnas secuenciales -> JSONB
    json_cols = {"actions", "times"}
    df_json = prepare_json_cols(df_norm, json_cols)

    # 3) Mapeo de tipos explícito
    dtypes = dtype_map_jsonb(df_json, json_cols)

    # 4) Crea esquema si no existe y escribe la tabla
    with engine.begin() as conn:
        if schema and schema != "public":
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    df_json.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists="replace",
        index=False,
        method="multi"
        chunksize=10_000,
        dtype=dtypes,
    )

    print(f"OK -> {schema}.{table_name}: {len(df_json):,} filas")



guardar_df_jsonb(df, "transactions")

