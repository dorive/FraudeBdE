from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="API para identificación de fraude")

class Transaccion(BaseModel):
    importe: float
    momento: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(tx: Transaccion):
    # Ejemplo de prueba
    prob = min(0.05 + tx.importe / 10000.0, 0.99)
    return {"probabilidad_fraude": prob}

