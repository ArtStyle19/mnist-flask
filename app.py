from flask import Flask, render_template, request, redirect, url_for, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

from preprocess import preprocess_image, to_base64
from tensorflow.keras.models import load_model


# ── config ────────────────────────────────
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

app   = Flask(__name__)
model = load_model("models/model.h5")       # ruta relativa al proyecto


def allowed(filename: str) -> bool:
    return filename and (filename.lower().rsplit(".", 1)[-1] in {
        ext.strip(".") for ext in ALLOWED_EXTENSIONS
    })


# ── routes ────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict/upload", methods=["POST"])
def predict_upload():
    file = request.files.get("image")
    if not (file and allowed(file.filename)):
        return redirect(url_for("index"))

    fname = secure_filename(file.filename)       # solo para mostrar
    pil   = Image.open(file.stream).convert("RGB")

    # ── preprocesar ───────────────────────
    x, proc_img = preprocess_image(pil, return_img=True)
    pred  = model.predict(x, verbose=0)[0]
    label = int(np.argmax(pred))
    score = f"{pred[label]*100:.2f}%"
    probs = [(i, float(p)) for i, p in enumerate(pred)]

    preview_b64 = to_base64(proc_img)

    return render_template(
        "result.html",
        filename=fname,
        label=label,
        score=score,
        probs=probs,
        preview=preview_b64
    )


@app.route("/predict/camera", methods=["POST"])
def predict_camera():
    """
    Recibe dataURL base‑64 → procesa → devuelve JSON.
    Trabaja enteramente en memoria; no guarda archivos.
    """
    try:
        data = request.get_json(force=True)
        if "image_data" not in data:
            return jsonify(success=False, error="No image_data found")

        # 1. decodificar captura ----------------------------------------------
        img_data  = data["image_data"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        pil       = Image.open(BytesIO(img_bytes)).convert("RGB")

        # 2. preprocesar + predecir -------------------------------------------
        x, proc_img = preprocess_image(pil, return_img=True)
        pred  = model.predict(x, verbose=0)[0]
        label = int(np.argmax(pred))
        score = f"{pred[label]*100:.2f}%"

        # 3. empaquetar respuesta ---------------------------------------------
        return jsonify(
            success=True,
            label=label,
            score=score,
            preview=to_base64(proc_img),
            probs=[float(p) for p in pred]
        )

    except Exception as e:
        return jsonify(success=False, error=str(e))


# ── run ────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

