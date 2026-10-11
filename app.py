import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_file, render_template

# Rutas de entorno: configurables por variable de entorno (RVC_DIR, FFMPEG_PATH),
# con los valores de esta máquina como predeterminados para que la app siga
# funcionando igual que siempre sin tener que configurar nada.
RVC_URL = os.environ.get("RVC_URL", "http://localhost:7865")
RVC_DIR = Path(os.environ.get("RVC_DIR", "/Users/franantartico/RVC-WebUI"))
RVC_PYTHON = RVC_DIR / "venv" / "bin" / "python"
TRAIN_RUNNER = RVC_DIR / "rvc_train_runner.py"
INFER_RUNNER = RVC_DIR / "rvc_infer_runner.py"
WEIGHTS_DIR = RVC_DIR / "assets" / "weights"
FFMPEG = os.environ.get("FFMPEG_PATH", "/Users/franantartico/pinokio/bin/miniconda/bin/ffmpeg")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = APP_DIR / "modelos"
for d in (DATASETS_DIR, UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".wav", ".mp3"}

# PYTORCH_ENABLE_MPS_FALLBACK: igual que en start.sh, para que cualquier
# operación no soportada en el backend MPS caiga a CPU en vez de fallar.
# OMP_NUM_THREADS/KMP_DUPLICATE_LIB_OK: en Apple Silicon, si el proceso ya
# tiene un tensor de PyTorch en MPS y luego faiss hace una búsqueda IVF
# multi-hilo (al usar el índice de retrieval en /api/convert), chocan dos
# runtimes OpenMP distintos y el proceso termina con SIGSEGV. Forzando un solo
# hilo se evita el choque sin coste apreciable (una sola consulta al índice
# por bloque de audio).
RVC_ENV = {
    **os.environ,
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "OMP_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "1",
}

app = Flask(__name__)


def sync_models_folder():
    """La carpeta modelos/ dentro de la app es la carpeta "de cara al usuario"
    para encontrar y gestionar los modelos entrenados. RVC solo puede cargar
    modelos desde su propia carpeta assets/weights, así que mantenemos las
    dos copiadas en ambos sentidos."""
    for weight_file in WEIGHTS_DIR.glob("*.pth"):
        target = MODELS_DIR / weight_file.name
        if not target.exists():
            shutil.copyfile(weight_file, target)
    for model_file in MODELS_DIR.glob("*.pth"):
        target = WEIGHTS_DIR / model_file.name
        if not target.exists():
            shutil.copyfile(model_file, target)


sync_models_folder()

# Estado en memoria de los entrenamientos en curso, para poder mostrar
# progreso/tiempo transcurrido en la interfaz mientras RVC procesa en background.
TRAIN_JOBS = {}


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return name or f"modelo_{uuid.uuid4().hex[:8]}"


def to_wav(src_path: Path, dst_path: Path):
    """Convierte cualquier wav/mp3 a wav 44.1kHz usando ffmpeg local (RVC no tiene
    ffmpeg en su PATH, así que también sirve para normalizar los .wav de entrada)."""
    subprocess.run(
        [FFMPEG, "-y", "-i", str(src_path), "-ar", "44100", "-ac", "1", str(dst_path)],
        check=True,
        capture_output=True,
    )


STEP_FROM_MARKER = {
    "RUNNER_STEP:preprocess": "preprocess",
    "RUNNER_STEP:features": "features",
    "RUNNER_STEP:training": "training",
    "RUNNER_STEP:index": "index",
    "RUNNER_STEP:done": "done",
}


def run_training(model_name: str, dataset_dir: Path, epochs: int):
    # Se llama directamente a las funciones de infer-web.py en un subproceso con el
    # propio intérprete/venv de RVC-WebUI: la API HTTP/Gradio de esta versión es
    # poco fiable para las funciones de entrenamiento (tipo generador).
    job = TRAIN_JOBS[model_name]
    try:
        proc = subprocess.Popen(
            [str(RVC_PYTHON), str(TRAIN_RUNNER), model_name, str(dataset_dir), str(epochs)],
            cwd=str(RVC_DIR),
            env=RVC_ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tail = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line in STEP_FROM_MARKER:
                job["step"] = STEP_FROM_MARKER[line]
            else:
                tail.append(line)
                tail[:] = tail[-50:]
                job["runner_log"] = "\n".join(tail)
        proc.wait()

        if weights_path(model_name).exists():
            sync_models_folder()
            job["status"] = "done"
        elif proc.returncode != 0:
            job["status"] = "error"
            job["error"] = "El proceso de entrenamiento de RVC terminó con error (ver log)."
        else:
            job["status"] = "error"
            job["error"] = "El entrenamiento terminó pero no se generó el archivo de modelo (dataset insuficiente)."
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


def weights_path(model_name: str) -> Path:
    return RVC_DIR / "assets" / "weights" / f"{model_name}.pth"


def tail_log(path: Path, n_chars=1500) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(errors="ignore")
        return text[-n_chars:]
    except Exception:
        return ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        r = requests.get(RVC_URL, timeout=3)
        return jsonify({"ok": r.status_code == 200})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/models")
def list_models():
    sync_models_folder()
    names = sorted(p.name for p in MODELS_DIR.glob("*.pth"))
    return jsonify({"models": names, "models_folder": str(MODELS_DIR)})


TRASH_DIR = MODELS_DIR / ".papelera"


@app.route("/api/models/<name>", methods=["DELETE"])
def delete_model(name):
    model_name = safe_name(Path(name).stem)
    weight_file = WEIGHTS_DIR / f"{model_name}.pth"
    model_file = MODELS_DIR / f"{model_name}.pth"
    if not weight_file.exists() and not model_file.exists():
        return jsonify({"ok": False, "error": "Ese modelo no existe."}), 404

    # Borrado seguro: el .pth se mueve a una papelera recuperable en vez de
    # eliminarse para siempre. El audio original de entrenamiento nunca se
    # borra (por si se quiere reentrenar); solo se limpian los archivos
    # intermedios pesados (features/f0), que se regeneran solos al reentrenar.
    TRASH_DIR.mkdir(exist_ok=True)
    if model_file.exists():
        shutil.move(str(model_file), str(TRASH_DIR / f"{model_name}_{int(time.time())}.pth"))
    weight_file.unlink(missing_ok=True)
    shutil.rmtree(RVC_DIR / "logs" / model_name, ignore_errors=True)
    TRAIN_JOBS.pop(model_name, None)

    return jsonify({"ok": True})


@app.route("/api/models/import", methods=["POST"])
def import_models():
    """Importa uno o varios .pth (por ejemplo encontrados al elegir una carpeta
    en el navegador) copiándolos a la carpeta modelos/ y a RVC para poder usarlos."""
    files = request.files.getlist("files")
    imported = []
    for f in files:
        if Path(f.filename).suffix.lower() != ".pth":
            continue
        name = safe_name(Path(f.filename).stem) + ".pth"
        f.save(MODELS_DIR / name)
        imported.append(name)

    if not imported:
        return jsonify({"ok": False, "error": "No se encontró ningún archivo .pth en esa carpeta."}), 400

    sync_models_folder()
    return jsonify({"ok": True, "imported": imported})


@app.route("/api/train", methods=["POST"])
def train():
    model_name = safe_name(request.form.get("model_name", ""))
    epochs = int(request.form.get("epochs", 20))
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "error": "No se subió ningún audio de entrenamiento."}), 400

    dataset_dir = DATASETS_DIR / model_name
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            continue
        file_id = uuid.uuid4().hex[:8]
        tmp_path = dataset_dir / f"raw_{file_id}{ext}"
        f.save(tmp_path)
        wav_path = dataset_dir / f"conv_{file_id}.wav"
        to_wav(tmp_path, wav_path)  # normaliza cualquier wav/mp3 de entrada
        tmp_path.unlink()

    n_wavs = len(list(dataset_dir.glob("*.wav")))
    if n_wavs == 0:
        return jsonify({"ok": False, "error": "Ningún archivo válido (.wav/.mp3) tras la conversión."}), 400

    TRAIN_JOBS[model_name] = {
        "status": "running",
        "step": "queued",
        "started_at": time.time(),
        "error": None,
        "n_files": n_wavs,
        "epochs": epochs,
        "runner_log": "",
    }
    threading.Thread(target=run_training, args=(model_name, dataset_dir, epochs), daemon=True).start()

    return jsonify({"ok": True, "model_name": model_name, "n_files": n_wavs})


STEP_LABELS = {
    "queued": "En cola…",
    "preprocess": "Procesando audios…",
    "features": "Extrayendo tono y características…",
    "training": "Entrenando el modelo (puede tardar bastante)…",
    "index": "Construyendo índice de búsqueda…",
}

# Rango de progreso (%) que ocupa cada fase dentro del entrenamiento completo.
# Son pesos aproximados (el preprocesado/extracción son rápidos, el
# entrenamiento por epochs es lo que domina el tiempo total).
STEP_RANGES = {
    "queued": (0, 2),
    "preprocess": (2, 15),
    "features": (15, 30),
    "training": (30, 92),
    "index": (92, 99),
    "done": (100, 100),
}

EPOCH_RE = re.compile(r"====> Epoch: (\d+)")


def estimate_train_percent(job) -> float:
    lo, hi = STEP_RANGES.get(job["step"], (0, 99))
    if job["step"] != "training":
        return float(lo)

    epochs_done = [int(m) for m in EPOCH_RE.findall(job.get("runner_log") or "")]
    total_epochs = max(job.get("epochs", 1), 1)
    frac = (max(epochs_done) / total_epochs) if epochs_done else 0.0
    frac = min(max(frac, 0.0), 1.0)
    return lo + frac * (hi - lo)


@app.route("/api/train/status")
def train_status():
    model_name = safe_name(request.args.get("model", ""))
    job = TRAIN_JOBS.get(model_name)
    if not job:
        return jsonify({"ok": False, "error": "No hay ningún entrenamiento con ese nombre."}), 404

    elapsed = round(time.time() - job["started_at"])
    log_dir = RVC_DIR / "logs" / model_name
    log_text = tail_log(log_dir / "preprocess.log")
    if job["step"] in ("features", "training", "index"):
        log_text = tail_log(log_dir / "extract_f0_feature.log") or log_text
    if job["step"] in ("training", "index"):
        log_text = job.get("runner_log") or log_text

    percent = 100.0 if job["status"] == "done" else estimate_train_percent(job)
    eta_seconds = None
    if job["status"] == "running" and percent > 2:
        eta_seconds = max(round(elapsed * (100 - percent) / percent), 0)

    return jsonify({
        "ok": True,
        "status": job["status"],
        "step": job["step"],
        "step_label": STEP_LABELS.get(job["step"], job["step"]),
        "elapsed_seconds": elapsed,
        "percent": round(percent, 1),
        "eta_seconds": eta_seconds,
        "error": job["error"],
        "log": log_text,
    })


def find_index_file(model_name: str) -> str:
    """Busca el índice de retrieval (.index) generado al entrenar el modelo.
    Vive en RVC-WebUI/logs/<model_name>/ y es lo que más ayuda a acercar el
    timbre generado al original (voces o instrumentos como guitarra)."""
    log_dir = RVC_DIR / "logs" / model_name
    if not log_dir.exists():
        return ""
    candidates = sorted(log_dir.glob("added_IVF*.index"))
    return str(candidates[0]) if candidates else ""


@app.route("/api/convert", methods=["POST"])
def convert():
    model_name = request.form.get("model_name", "")
    pitch = int(request.form.get("pitch", 0))
    index_rate = float(request.form.get("index_rate", 0.6))
    f = request.files.get("file")
    if not model_name:
        return jsonify({"ok": False, "error": "Falta model_name."}), 400
    if not f:
        return jsonify({"ok": False, "error": "No se subió audio a clonar."}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": "Formato no soportado, usa wav o mp3."}), 400

    job_id = uuid.uuid4().hex[:8]
    raw_path = UPLOADS_DIR / f"src_{job_id}{ext}"
    f.save(raw_path)
    wav_path = UPLOADS_DIR / f"conv_{job_id}.wav"
    to_wav(raw_path, wav_path)
    raw_path.unlink()

    sync_models_folder()
    candidates = sorted(WEIGHTS_DIR.glob(f"*{model_name}*.pth"))
    if not candidates:
        return jsonify({"ok": False, "error": f"Modelo '{model_name}' no encontrado. ¿Ya terminó de entrenar?"}), 404
    model_file = candidates[0].name

    out_name = f"clonado_{model_name}_{job_id}.wav"
    out_path = OUTPUTS_DIR / out_name

    index_file = find_index_file(model_name)
    effective_index_rate = index_rate if index_file else 0.0

    # Se invoca directamente la función de RVC en un subproceso con su propio
    # intérprete/venv (la API HTTP/Gradio de esta versión no es fiable).
    proc = subprocess.run(
        [
            str(RVC_PYTHON), str(INFER_RUNNER), model_file, str(wav_path), str(pitch), str(out_path),
            index_file, str(effective_index_rate),
        ],
        cwd=str(RVC_DIR),
        env=RVC_ENV,
        capture_output=True,
        text=True,
    )
    last_line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("RESULT_")), "")
    if proc.returncode != 0 or not last_line.startswith("RESULT_OK"):
        error_msg = last_line.split(":", 1)[1] if ":" in last_line else (proc.stderr[-800:] or "Error desconocido")
        return jsonify({"ok": False, "error": error_msg}), 500

    info = last_line.split(":", 1)[1] if ":" in last_line else "Listo."
    return jsonify({"ok": True, "download_url": f"/api/download/{out_name}", "info": info})


@app.route("/api/download/<name>")
def download(name):
    path = OUTPUTS_DIR / name
    if not path.exists():
        return jsonify({"ok": False, "error": "Archivo no encontrado"}), 404
    return send_file(path, as_attachment=True, download_name=name, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True, use_reloader=False)
