import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from preprocess import preprocess_image         # tu función de limpieza (28×28)

# ── configuración de carpetas ────────────────────────────────────────────
APP_ROOT   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# ── iniciar Flask y cargar el modelo ─────────────────────────────────────
app   = Flask(__name__)
model = load_model(os.path.join(APP_ROOT, "./models/model.h5"))

def allowed(filename: str) -> bool:
    """Comprueba la extensión."""
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

# ── rutas ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file and allowed(file.filename):
            # 1) guardar el archivo
            fname = secure_filename(file.filename)
            path  = os.path.join(UPLOAD_DIR, fname)
            file.save(path)

            # 2) preprocesar y predecir
            x       = preprocess_image(path)           # (1,28,28,1)
            pred    = model.predict(x, verbose=0)[0]   # (10,)
            label   = int(np.argmax(pred))
            probs   = [(i, float(p)) for i, p in enumerate(pred)]
            top_pct = f"{pred[label]*100:.2f}%"

            # 3) mostrar resultado
            return render_template(
                "result.html",
                filename=fname,
                label=label,
                score=top_pct,
                probs=probs
            )
        # archivo no válido → recarga
        return redirect(url_for("index"))
    return render_template("index.html")

# ── ejecutar ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 0.0.0.0 → accesible desde tu celular en la misma red
    app.run(host="0.0.0.0", port=5000, debug=True)
