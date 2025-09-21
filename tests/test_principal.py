import pickle
import os
import pytest

def test_encoder_embeddings_pkl_exists():
    path = os.path.join("app", "modelos", "encoder_embeddings.pkl")
    assert os.path.exists(path), f"El archivo {path} no existe"

def test_encoder_embeddings_pkl_keys():
    path = os.path.join("app", "modelos", "encoder_embeddings.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Claves que deberían estar en el pkl
    expected_keys = {"Z_train", "Z_val", "Z_test"}

    assert expected_keys.issubset(data.keys()), (
        f"Faltan claves en el pkl. Encontradas: {list(data.keys())}"
    )

