import os, sys
import pytest
import torch
import io
import pickle
sys.path.append("..")
from app.utils.torch_utils import set_seed, CPU_Unpickler

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

def test_encoder_results_pkl_exists():
    path = os.path.join("app", "modelos", "encoder_results.pkl")
    assert os.path.exists(path), f"El archivo {path} no existe"

def test_encoder_results_pkl_keys():
    path = os.path.join("app", "modelos", "encoder_results.pkl")
    with open(path, "rb") as f:
        data = CPU_Unpickler(f).load()

    expected_keys = {
        "model",
        "Z_train", "p_seq_train",
        "Z_val", "p_seq_val",
        "Z_test", "p_seq_test",
    }

    assert expected_keys.issubset(data.keys()), (
        f"Faltan claves en el pkl. Encontradas: {list(data.keys())}"
    )
