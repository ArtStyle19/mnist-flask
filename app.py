from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

from preprocess import preprocess_image, to_base64      # ⬅️  nuevo
from tensorflow.keras.models import load_model


# ── config ────────────────────────────────
APP_ROOT   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

app   = Flask(__name__)
model = load_model(os.path.join(APP_ROOT, "models", "model.h5"))


def allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


# ── routes ────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict/upload", methods=["POST"])
def predict_upload():
    file = request.files.get("image")
    if not (file and allowed(file.filename)):
        return redirect(url_for("index"))

    fname = secure_filename(file.filename)
    path  = os.path.join(UPLOAD_DIR, fname)
    file.save(path)

    # ── preprocesar ───────────────────────
    x, proc_img = preprocess_image(path, return_img=True)
    pred  = model.predict(x, verbose=0)[0]
    label = int(np.argmax(pred))
    score = f"{pred[label]*100:.2f}%"
    probs = [(i, float(p)) for i, p in enumerate(pred)]

    preview_b64 = to_base64(proc_img)       # ⬅️  ¡nuevo!

    return render_template(
        "result.html",
        filename=fname,
        label=label,
        score=score,
        probs=probs,
        preview=preview_b64                 # ⬅️  ¡nuevo!
    )


@app.route("/predict/camera", methods=["POST"])
def predict_camera():
    try:
        data = request.get_json(force=True)
        if "image_data" not in data:
            return jsonify(success=False, error="No image_data found")

        # 1. decodificar captura ------------------------------------------------
        img_data  = data["image_data"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img       = Image.open(BytesIO(img_bytes)).convert("RGB")

        path = os.path.join(UPLOAD_DIR, "captured.png")  # se sobre‑escribe
        img.save(path)

        # 2. preprocesar + predecir --------------------------------------------
        x, proc_img = preprocess_image(path, return_img=True)
        pred  = model.predict(x, verbose=0)[0]           # array de 10 floats
        label = int(np.argmax(pred))
        score = f"{pred[label]*100:.2f}%"

        # 3. empaquetar respuesta ----------------------------------------------
        preview_b64 = to_base64(proc_img)
        return jsonify(
            success=True,
            label=label,
            score=score,
            preview=preview_b64,
            probs=[float(p) for p in pred]     #  ⬅️  listado de 10 números
        )

    except Exception as e:
        return jsonify(success=False, error=str(e))


# @app.route("/predict/camera", methods=["POST"])
# def predict_camera():
#     """
#     Recibe un dataURL base‑64 del navegador, lo preprocesa y
#     devuelve JSON con la predicción y una vista previa 28×28.
#     """
#     try:
#         data = request.get_json(force=True)
#         if "image_data" not in data:
#             return jsonify(success=False, error="No image_data found")
#
#         # --- 1. decodificar captura ---
#         img_data  = data["image_data"].split(",")[1]
#         img_bytes = base64.b64decode(img_data)
#         img       = Image.open(BytesIO(img_bytes)).convert("RGB")
#
#         # guardar temporal (opcional, útil para depurar)
#         path = os.path.join(UPLOAD_DIR, "captured.png")
#         img.save(path)
#
#         # --- 2. preprocesar + predecir ---
#         x, proc_img = preprocess_image(path, return_img=True)
#         pred  = model.predict(x, verbose=0)[0]
#         label = int(np.argmax(pred))
#         score = f"{pred[label]*100:.2f}%"
#
#         # --- 3. preparar respuesta ---
#         preview_b64 = to_base64(proc_img)
#         return jsonify(
#             success=True,
#             label=label,
#             score=score,
#             preview=preview_b64
#         )
#
#     except Exception as e:
#         return jsonify(success=False, error=str(e))


# ── run ────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
