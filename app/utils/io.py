from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

def leer_raw_csv(name: str) -> pd.DataFrame:
    """
    Lee un CSV original desde data/raw
    INPUT:
        name: nombre del fichero CSV (ej: "creditcard.csv")
    OUTPUT:
        df: DataFrame con los datos leídos
    """
    return pd.read_csv(DATA_DIR / "orig" / name)



def guardar_parquet(df: pd.DataFrame, name: str, sub="procesado"):
    """
    Guarda un DataFrame en formato parquet en data/{sub}
    INPUT:
        df: DataFrame a guardar
        name: nombre del fichero (sin extensión)
        sub: subdirectorio dentro de data (default: "procesado")
    """
    out = DATA_DIR / sub / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
