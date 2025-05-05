# preprocess.py – versión in‑memory
import base64
from io import BytesIO
from typing import Union

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
    """Convierte la imagen 28×28 uint8 a <img src="data:...">."""
    pil = Image.fromarray(img28x28)        # modo “L”
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ───────────── preprocesado principal ───────────
def preprocess_image(
    img_or_path: Union[str, Image.Image],
    *,
    return_img: bool = False
):
    """
    Recibe una ruta **o** un objeto PIL.Image y devuelve la imagen
    normalizada (1,28,28,1).  Opciónally, devuelve también la 28×28 uint8.
    """
    # 1. abrir imagen ----------------------------------------------------------
    if isinstance(img_or_path, Image.Image):
        pil = img_or_path.convert("RGB")
    elif isinstance(img_or_path, str):
        pil = Image.open(img_or_path).convert("RGB")
    else:
        raise TypeError("preprocess_image: espera ruta str o PIL.Image")

    gray = np.array(ImageOps.grayscale(pil))

    # 2. realce local de contraste (CLAHE) -------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # 3. binarización Otsu global ---------------------------------------------
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. adaptativo si quedó vacía --------------------------------------------
    if _is_empty(bin_img):
        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2
        )

    # 5. umbral fijo si aún vacía ---------------------------------------------
    if _is_empty(bin_img):
        _, bin_img = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # 6. resize + normalizar ---------------------------------------------------
    resized = cv2.resize(
        bin_img, (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_AREA
    )
    norm = resized.astype("float32") / 255.0
    norm = norm.reshape(1, TARGET_SIZE, TARGET_SIZE, 1)

    return (norm, resized) if return_img else norm
