import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv
import joblib
import numpy as np

# Cargar variables de entorno
load_dotenv()

# Rutas de modelos
MODELS_DIR = Path(__file__).resolve().parents[1] / "modelos"
MODEL_PATH = MODELS_DIR / "model.pkl"
META_PATH = MODELS_DIR / "metadata.json"

# Comprobar que existen los modelos
if not MODEL_PATH.exists() or not META_PATH.exists():
    raise RuntimeError("No se encuentra model.pkl o metadata.json en app/modelos/. Exporta el modelo desde el Día 4.")

# Carga modelos en memoria
CALIBRADOR = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    METADATA = json.load(f)

FEATURE_ORDER: List[str] = METADATA.get("feature_order", [])
UMBRAL_ALERTA = float(os.getenv("UMBRAL_ALERTA", "0.8"))

app = FastAPI(title="API de Scoring de Fraude", version=METADATA.get("version", "0.0.0"))

class Transaccion(BaseModel):
    """
    Modelo de datos para la transacción a evaluar.
    """
    
    atributos: dict[str, float] = Field(..., description="Diccionario con las columnas requeridas para el modelo")

    @model_validator(mode="after")
    def validar_features(self):
        """
        Valida que se proporcionen todas las columnas requeridas.
        """

        feats = self.atributos or {}
        if not FEATURE_ORDER:
            if not _try_load_model():
                raise ValueError("El modelo no está disponible en el servidor.")
        faltantes = [c for c in FEATURE_ORDER if c not in feats]
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas: {faltantes}")
        
        # Asegurar que los atributos están en el orden correcto y son float
        self.atributos = {k: float(feats[k]) for k in FEATURE_ORDER if k in feats}
        return self

class RespuestaScore(BaseModel):
    """
    Modelo de datos para la respuesta del scoring.
    """ 
    probabilidad_fraude: float
    flag_alerta: bool
    umbral: float
    modelo_version: str
    latencia_ms: float


@app.get("/health")
def health():
    """ 
    Endpoint para verificar que la API está funcionando.
    """
    return {"estado": "ok"}


@app.get("/metadata")
def metadata():
    """
    Endpoint para obtener metadata del modelo.
    """

    return {
        "modelo": METADATA.get("modelo"),
        "version": METADATA.get("version"),
        "fecha_entrenamiento": METADATA.get("fecha_entrenamiento"),
        "target": METADATA.get("target"),
        "num_features": len(FEATURE_ORDER),
        "features": FEATURE_ORDER[:10] + (["..."] if len(FEATURE_ORDER) > 10 else [])
    }


@app.post("/score", response_model=RespuestaScore)
def score(tx: Transaccion):
    """
    Endpoint para obtener el score de una transacción.
    """
    
    t0 = time.perf_counter()
    try:
        x = np.array([[tx.atributos[c] for c in FEATURE_ORDER]], dtype=float)
        # El calibrador expone predict_proba si el base estimator lo soporta
        prob = float(CALIBRADOR.predict_proba(x)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error de inferencia: {e}")

    lat_ms = (time.perf_counter() - t0) * 1000.0
    return RespuestaScore(
        probabilidad_fraude=prob,
        flag_alerta=prob >= UMBRAL_ALERTA,
        umbral=UMBRAL_ALERTA,
        modelo_version=METADATA.get("version", "desconocida"),
        latencia_ms=lat_ms
    )


