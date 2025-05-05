# preprocess.py – versión "anti‑pantalla‑blanca" + utilidades extra
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

TARGET_SIZE = 28
EMPTY_TOL   = 0.01           # <1 % píxeles → se considera “vacío”


# ───────────────── utilidades ──────────────────
def _is_empty(img: np.ndarray) -> bool:
    """True si la imagen binaria está (casi) vacía."""
    return np.count_nonzero(img) / img.size < EMPTY_TOL


def to_base64(img28x28: np.ndarray) -> str:
    """
    Convierte una imagen 28×28 en escala de grises (dtype uint8)
    en una cadena base‑64 lista para usar como <img src="data:...">.
    """
    pil = Image.fromarray(img28x28)        # modo “L”
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ───────────── preprocesado principal ───────────
def preprocess_image(path: str, *, return_img: bool = False):
    """
    Limpia una foto de un dígito manuscrito y la deja lista
    para un modelo tipo MNIST (shape (1,28,28,1)).

    Parámetros
    ----------
    path : str
        Ruta al archivo original.
    return_img : bool, opcional (False)
        Si es True, devuelve también la imagen 28×28 uint8.

    Returns
    -------
    np.ndarray
        Imagen normalizada con shape (1,28,28,1) y rango [0,1].
    np.ndarray   (solo si return_img=True)
        Imagen binaria 28 × 28 uint8 (0–255).
    """
    # 1. cargar en RGB → gris
    pil  = Image.open(path).convert("RGB")
    gray = np.array(ImageOps.grayscale(pil))

    # 2. realce local de contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # 3. primera pasada binaria (Otsu global)
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. si quedó vacía, prueba adaptativo
    if _is_empty(bin_img):
        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2
        )

    # 5. si sigue vacía, umbral fijo suavizado
    if _is_empty(bin_img):
        _, bin_img = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # 6. resize a 28×28 y normalizar
    resized = cv2.resize(
        bin_img, (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_AREA
    )
    norm = resized.astype("float32") / 255.0
    norm = norm.reshape(1, TARGET_SIZE, TARGET_SIZE, 1)

    # opcional: guardar para depurar
    # cv2.imwrite("debug.png", resized)

    return (norm, resized) if return_img else norm
