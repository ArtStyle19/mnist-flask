# preprocess.py – versión "anti-pantalla-blanca"
import cv2
import numpy as np
from PIL import Image, ImageOps

TARGET_SIZE = 28
EMPTY_TOL   = 0.01   # <1 % píxeles → se considera “vacío”

def _is_empty(img: np.ndarray) -> bool:
    return np.count_nonzero(img) / img.size < EMPTY_TOL

def preprocess_image(path) -> np.ndarray:
    """Limpia una foto de un dígito manuscrito y la deja en (1,28,28,1)."""
    # 1. cargar y gris
    pil = Image.open(path).convert("RGB")
    gray = np.array(ImageOps.grayscale(pil))

    # 2. realce de contraste local (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # 3. — primera pasada: Otsu global —
    _, bin_img = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. ¿quedó vacía? prueba adaptativo
    if _is_empty(bin_img):
        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2
        )

    # 5. ¿aún vacía? baja ligeramente el umbral fijo y reintenta
    if _is_empty(bin_img):
        _, bin_img = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # 6. redimensionar y normalizar
    resized = cv2.resize(bin_img, (TARGET_SIZE, TARGET_SIZE),
                         interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0


    cv2.imwrite("debug.png", resized)
    return norm.reshape(1, TARGET_SIZE, TARGET_SIZE, 1)
