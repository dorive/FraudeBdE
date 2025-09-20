# -*- coding: utf-8 -*-
"""
ActionTransformer: encoder basado en Transformer para secuencias de acciones.

Resumen
-------
Este módulo implementa:
1) Un modelo ActionTransformer (Transformer encoder) que:
   - Embebe (embedding) IDs de acciones.
   - Preprende un token [CLS] para obtener una representación global de la secuencia.
   - Aplica codificación posicional aprendible (learnable positional embeddings).
   - Usa enmascarado de padding para ignorar posiciones vacías.
   - Predice una probabilidad (sigmoid) y expone el embedding z_cls del [CLS].

2) Un Dataset ligero para tensores de IDs y etiquetas.

3) Un bucle de entrenamiento con early stopping por AUPRC en validación.

4) Una función utilitaria para extraer:
   - Z: matriz de embeddings z_cls por secuencia.
   - P: probabilidades predichas.

Notas
-----
- Se asume que el ID 0 es padding (PAD).
- La métrica de validación es Average Precision (AUPRC), útil en datasets desbalanceados.
- La pérdida es Binary Cross-Entropy (BCELoss) porque el problema es binario.

Autor: (tu nombre)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
import numpy as np
from app.utils.torch_utils import set_seed

# -----------------------------
# 1) Reproducibilidad
# -----------------------------
RANDOM_STATE = 42
set_seed(RANDOM_STATE)


class ActionTransformer(nn.Module):
    """
    Encoder secuencial basado en Transformer para clasificación binaria a partir
    de secuencias de acciones (token IDs).

    Parámetros
    ----------
    n_actions : int
        Número de acciones distintas (tamaño del vocabulario de acciones).
        Se reservará un índice adicional para UNK/padding de seguridad.
    d_model : int, opcional (por defecto 128)
        Dimensión del embedding y de las capas internas del Transformer.
    max_len : int, opcional (por defecto 256)
        Longitud máxima de la secuencia (para la codificación posicional).
    n_heads : int, opcional (por defecto 4)
        Número de cabezas de atención (multi-head attention).
    n_layers : int, opcional (por defecto 3)
        Número de capas del TransformerEncoder.
    p_drop : float, opcional (por defecto 0.1)
        Dropout usado en el encoder y en la cabecera (head).

    Atributos
    ---------
    emb : nn.Embedding
        Capa de embedding para mapear IDs de acciones a vectores d_model.
        padding_idx=0 indica que el índice 0 es PAD y no se entrena.
    cls : nn.Parameter
        Vector aprendible para el token especial [CLS], tamaño (1,1,d_model).
    encoder : nn.TransformerEncoder
        Bloque encoder con n_layers de TransformerEncoderLayer.
    head : nn.Sequential
        MLP final que mapea z_cls -> probabilidad (sigmoid) de clase positiva.
    pos : nn.Parameter
        Embeddings posicionales aprendibles, tamaño (1, max_len+1, d_model),
        el +1 es para incluir la posición del token [CLS].

    Forward
    -------
    x_ids : torch.LongTensor, shape (B, L)
        Mini-lote de secuencias de IDs (B=batch, L=longitud).
        Se asume que 0 es PAD.

    Devuelve
    --------
    p : torch.FloatTensor, shape (B,)
        Probabilidades (0..1) tras sigmoid.
    z_cls : torch.FloatTensor, shape (B, d_model)
        Embedding del token [CLS], representación global de la secuencia.
    """

    def __init__(self, n_actions, d_model=128, max_len=256, n_heads=4, n_layers=3, p_drop=0.1):
        super().__init__()
        self.max_len = max_len

        # Embedding de tokens de acciones. +1 para reservar un índice extra (p.ej., UNK).
        self.emb = nn.Embedding(n_actions + 1, d_model, padding_idx=0)  # 0 = PAD

        # Token especial [CLS] aprendible: agrega un resumen global al inicio de la secuencia.
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Definición de la capa base del Transformer encoder:
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,   # tamaño del MLP interno del Transformer
            dropout=p_drop,
            batch_first=True               # entradas como (B, L, d_model)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Cabecera (head) de clasificación binaria que opera sobre z_cls.
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, 1)              # salida escalar por muestra
        )

        # Embedding posicional aprendible (learnable positional embedding).
        # +1 para la posición del [CLS] al comienzo.
        self.pos = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.01)

    def forward(self, x_ids):
        """
        Aplica embedding + [CLS] + posición -> Transformer encoder -> head (sigmoid).

        Pasos
        -----
        1) Embedding de tokens.
        2) Prependemos [CLS] y sumamos embeddings posicionales.
        3) Creamos máscara de padding (True donde hay PAD) para ignorar esas posiciones.
        4) Pasamos por el Transformer encoder.
        5) Extraemos z_cls (posición 0) y calculamos probabilidad con la head.

        Parámetros
        ----------
        x_ids : torch.LongTensor, shape (B, L)

        Returns
        -------
        p : torch.FloatTensor, shape (B,)
            Probabilidad en [0,1] tras sigmoid.
        z_cls : torch.FloatTensor, shape (B, d_model)
            Representación global de la secuencia (embedding del [CLS]).
        """
        B, L = x_ids.size()

        # Expandimos el token [CLS] al tamaño del batch: (B, 1, d_model)
        cls_tok = self.cls.expand(B, 1, -1)

        # Embedding de las secuencias: (B, L, d_model)
        x = self.emb(x_ids)

        # Concatenamos [CLS] delante: (B, L+1, d_model) y sumamos posición
        x = torch.cat([cls_tok, x], dim=1) + self.pos[:, :L + 1, :]

        # Máscara de padding: True donde hay PAD (ID==0) => el encoder lo ignorará
        pad_mask = (x_ids == 0)  # (B, L)
        # Añadimos falsa en la posición del [CLS] (nunca se enmascara)
        pad_mask = torch.cat([torch.zeros(B,1, dtype=torch.bool, device=x.device), pad_mask], dim=1)

        # Encoder Transformer con máscara de padding
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L+1, d_model)

        # Tomamos la representación del [CLS] (posición 0)
        z_cls = h[:, 0, :]  # (B, d_model)

        # Head de clasificación + sigmoid -> probabilidad
        p = torch.sigmoid(self.head(z_cls)).squeeze(-1)  # (B,)

        return p, z_cls


class SeqDataset(Dataset):
    """
    Dataset mínimo para secuencias de IDs y etiquetas binarias.

    Parámetros
    ----------
    x_ids : np.ndarray o array-like, shape (N, L)
        Matriz de IDs de tokens por muestra.
    y : np.ndarray o array-like, shape (N,)
        Etiquetas (0/1).

    Notas
    -----
    - __getitem__ devuelve tensores PyTorch listos para el DataLoader.
    """

    def __init__(self, x_ids, y):
        self.x = x_ids
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # Devuelve (x_i, y_i) como tensores; y como float32 para BCELoss.
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i], dtype=torch.float32)


def train_encoder(
    x_train_ids, y_train,
    x_val_ids, y_val,
    n_actions, device, max_len,
    epochs=10, bs=256, lr=2e-4
):
    """
    Entrena EncoderSecuencial con early stopping por AUPRC en validación.

    Parámetros
    ----------
    x_train_ids : np.ndarray, shape (N_tr, L)
    y_train     : np.ndarray, shape (N_tr,)
    x_val_ids   : np.ndarray, shape (N_va, L)
    y_val       : np.ndarray, shape (N_va,)
    n_actions   : int
        Tamaño del vocabulario de acciones.
    device      : torch.device
        'cuda' o 'cpu'.
    max_len     : int
        Longitud máxima de secuencia (debe casar con los tensores).
    epochs      : int
        Número de épocas de entrenamiento.
    bs          : int
        Batch size.
    lr          : float
        Learning rate del AdamW.

    Returns
    -------
    model : EncoderSecuencial
        Modelo con los mejores pesos según AUPRC en validación.
    """
    # Instanciamos el modelo y los componentes de entrenamiento
    model = EncoderSecuencial(n_actions=n_actions, max_len=max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCELoss()

    # DataLoaders (shuffle solo en train)
    tr_dl = DataLoader(SeqDataset(x_train_ids, y_train), batch_size=bs, shuffle=True,  num_workers=2)
    va_dl = DataLoader(SeqDataset(x_val_ids,   y_val),   batch_size=bs, shuffle=False, num_workers=2)

    best_state, best_pr = None, -1.0

    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")

        # -----------------------------
        # Fase de entrenamiento
        # -----------------------------
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            p, _ = model(xb)                 # forward
            loss = crit(p, yb)               # BCELoss
            opt.zero_grad()
            loss.backward()
            opt.step()

        # -----------------------------
        # Evaluación en validación
        # -----------------------------
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                p, _ = model(xb)
                ps.append(p.detach().cpu().numpy())
                ys.append(yb.numpy())

        # Métrica de early stopping: Average Precision (AUPRC)
        ap = average_precision_score(np.concatenate(ys), np.concatenate(ps))

        # Guardamos el mejor estado (en CPU para ahorrar VRAM)
        if ap > best_pr:
            best_pr = ap
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {ep+1}/{epochs} | AUPRC(val)={ap:.4f}")

    # Cargamos los mejores pesos y reportamos la mejor AUPRC
    model.load_state_dict(best_state)
    print(f"Best AUPRC(val)={best_pr:.4f}")
    return model


def extract_seq_features(model, x_ids, device, bs=512):
    """
    Extrae embeddings de secuencia y probabilidades usando un modelo entrenado.

    Parámetros
    ----------
    model : EncoderSecuencial
        Modelo ya entrenado.
    x_ids : np.ndarray, shape (N, L)
        Secuencias de IDs (0=PAD).
    device : torch.device
        'cuda' o 'cpu'.
    bs : int, opcional (por defecto 512)
        Tamaño de batch para la inferencia.

    Returns
    -------
    Z : np.ndarray, shape (N, d_model)
        Embeddings z_cls por muestra (representación global).
    P : np.ndarray, shape (N,)
        Probabilidades predichas por el modelo.
    """
    # Creamos un DataLoader "dummy" con etiquetas cero (no se usan)
    dl = DataLoader(SeqDataset(x_ids, np.zeros(len(x_ids))), batch_size=bs, shuffle=False)

    Z, P = [], []
    model.eval()
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            p, z = model(xb)
            Z.append(z.cpu().numpy())
            P.append(p.cpu().numpy())

    # Apilamos resultados de todos los batches
    return np.vstack(Z), np.concatenate(P)
