import os
import io
import random
import torch
import pickle
import numpy as np


class CPU_Unpickler(pickle.Unpickler):
    """
    Custom Unpickler para cargar objetos de PyTorch forzando
    el almacenamiento en CPU en lugar de GPU.

    Motivación
    ----------
    - Cuando se guardan modelos o tensores con pickle en una máquina con GPU,
      el objeto puede quedar asociado a un dispositivo CUDA.
    - Si luego intentamos cargarlo en una máquina sin GPU, da error.
    - Esta clase reimplementa `find_class` para interceptar la llamada
      a `torch.storage._load_from_bytes` y forzar `map_location='cpu'`.

    Uso
    ---
    obj = CPU_Unpickler(open("fichero.pkl", "rb")).load()
    """

    def find_class(self, module, name):
        # Caso especial: carga de tensores almacenados con torch.storage
        if module == 'torch.storage' and name == '_load_from_bytes':
            # Redirigimos a torch.load con map_location='cpu'
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            # Para el resto, usar el comportamiento estándar
            return super().find_class(module, name)


def set_seed(seed):
    """
    Fija la semilla (seed) en todas las librerías relevantes para garantizar
    reproducibilidad en experimentos de deep learning.

    Parámetros
    ----------
    seed : int
        Valor entero usado como semilla global.

    Acciones
    --------
    - random.seed(seed)       : fija la semilla del módulo random de Python.
    - np.random.seed(seed)    : fija la semilla de NumPy.
    - torch.manual_seed(seed) : fija la semilla para tensores en CPU.
    - torch.cuda.manual_seed  : fija la semilla para tensores en GPU.
    - torch.backends.cudnn.*  : fuerza ejecución determinista en cuDNN.
    - os.environ[...]         : variables de entorno para reproducibilidad
                                (CUBLAS y PYTHONHASHSEED).
    - torch.use_deterministic_algorithms(True)
        asegura que todas las operaciones en PyTorch sean deterministas
        (aunque pueda ser más lento).
    """

    # Librerías estándar
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU y GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Configuración determinista para cuDNN (backend de GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Variables de entorno adicionales
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # reproducibilidad en CUBLAS
    os.environ['PYTHONHASHSEED'] = str(seed)           # reproducibilidad en hash aleatorios de Python

    # PyTorch fuerza algoritmos deterministas
    torch.use_deterministic_algorithms(True)


