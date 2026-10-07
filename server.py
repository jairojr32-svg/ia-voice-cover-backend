"""
Sample Box — motor de separação de vocal (Demucs) para Hugging Face Spaces (Docker, CPU grátis).
Baseado em backend/server.py; diferenças: device CPU, CORS por variável de ambiente, porta 7860.

Endpoints:
  GET  /health            → status
  POST /analyze           → BPM/tom (librosa)   [corpo = bytes do áudio, header x-filename]
  POST /separate          → vocal isolado (WAV) [corpo = bytes do áudio, header x-filename]
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

app = FastAPI(title="Sample Box — Vocal Extractor (HF)")

# CORS: domínios permitidos vêm de ALLOWED_ORIGINS (separados por vírgula). "*" libera geral.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
_origins = ["*"] if _origins_env in ("", "*") else [o.strip() for o in _origins_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "htdemucs"
MAX_BYTES = 50 * 1024 * 1024  # 50 MB

_lock: "asyncio.Lock | None" = None


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def _run_demucs(src: str, outdir: str) -> None:
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", MODEL, "-d", "cpu", "-o", outdir, src]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "demucs falhou")


def _trim(src: str, seconds: int) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    dst = src + f".prev{seconds}.wav"
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", src, "-t", str(seconds), "-ac", "2", "-ar", "44100", dst],
        capture_output=True, text=True,
    )
    return dst if proc.returncode == 0 and os.path.exists(dst) else src


@app.get("/health")
def health():
    return {"ok": True, "device": "cpu", "model": MODEL}


# --- BPM / Tom (librosa) -----------------------------------------------------
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTES_EN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTES_PT = ["Dó", "Dó#", "Ré", "Ré#", "Mi", "Fá", "Fá#", "Sol", "Sol#", "Lá", "Lá#", "Si"]


def _corr(a, b) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((a[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((b[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else 0.0


def _analyze(path: str) -> dict:
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True, duration=90)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(tempo)))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = [float(v) for v in chroma.mean(axis=1)]
    best = {"score": -2.0, "i": 0, "scale": "major"}
    for i in range(12):
        rot = profile[i:] + profile[:i]
        for scale, ks in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
            score = _corr(rot, ks)
            if score > best["score"]:
                best = {"score": score, "i": i, "scale": scale}
    i, scale = best["i"], best["scale"]
    return {
        "bpm": bpm,
        "key": f"{_NOTES_EN[i]}{'' if scale == 'major' else 'm'}",
        "scale": scale,
        "label_pt": f"{_NOTES_PT[i]} {'maior' if scale == 'major' else 'menor'}",
        "confidence": round(max(0.0, best["score"]), 2),
    }


@app.post("/analyze")
async def analyze(request: Request):
    from urllib.parse import unquote

    data = await request.body()
    filename = unquote(request.headers.get("x-filename", "audio.mp3"))
    if not data:
        raise HTTPException(400, "Arquivo vazio (0 bytes recebidos).")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Arquivo acima de 50 MB.")
    work = tempfile.mkdtemp(prefix="samplebox_an_")
    ext = os.path.splitext(filename)[1] or ".mp3"
    src = os.path.join(work, uuid.uuid4().hex + ext)
    with open(src, "wb") as f:
        f.write(data)
    try:
        return await asyncio.to_thread(_analyze, src)
    except Exception as e:
        raise HTTPException(500, f"Falha ao analisar: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/separate")
async def separate(request: Request):
    from urllib.parse import unquote

    data = await request.body()
    filename = unquote(request.headers.get("x-filename", "audio.mp3"))
    print(f"[separate] recebido {len(data)} bytes · nome={filename}", file=sys.stderr)
    if not data:
        raise HTTPException(400, "Arquivo vazio (0 bytes recebidos).")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Arquivo acima de 50 MB.")
    work = tempfile.mkdtemp(prefix="samplebox_")
    name = uuid.uuid4().hex
    ext = os.path.splitext(filename)[1] or ".mp3"
    src = os.path.join(work, name + ext)
    with open(src, "wb") as f:
        f.write(data)
    try:
        preview = int(request.headers.get("x-preview-seconds", "0"))
    except ValueError:
        preview = 0
    demux_src = src
    if preview > 0:
        demux_src = await asyncio.to_thread(_trim, src, preview)
    out = os.path.join(work, "out")
    stem = os.path.splitext(os.path.basename(demux_src))[0]
    t0 = time.monotonic()
    async with _get_lock():
        try:
            await asyncio.to_thread(_run_demucs, demux_src, out)
        except RuntimeError as e:
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(500, f"Falha ao separar: {e}")
    print(f"[separate] concluído em {time.monotonic() - t0:.0f}s", file=sys.stderr)
    vocals = os.path.join(out, MODEL, stem, "vocals.wav")
    if not os.path.exists(vocals):
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, "Vocal não gerado.")
    cleanup = BackgroundTask(shutil.rmtree, work, ignore_errors=True)
    return FileResponse(vocals, media_type="audio/wav", filename="vocal.wav", background=cleanup)


# --- Singing Voice Conversion (RVC) -----------------------------------------

RVC_PYTHON = sys.executable
RVC_SCRIPT = "rvc_infer.py"
RVC_MODELS_DIR = "models"


def _list_voices() -> "list[str]":
    """Available trained models = subfolders of models/ that contain a .pth."""
    if not os.path.isdir(RVC_MODELS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RVC_MODELS_DIR)):
        folder = os.path.join(RVC_MODELS_DIR, name)
        if os.path.isdir(folder) and any(f.endswith(".pth") for f in os.listdir(folder)):
            out.append(name)
    return out


def _run_rvc(model: str, src: str, dst: str, pitch: int, device: str) -> None:
    cmd = [
        RVC_PYTHON, RVC_SCRIPT,
        "--model", model,
        "--input", src,
        "--output", dst,
        "--pitch", str(pitch),
        "--device", device,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # AQUI ESTÁ A MUDANÇA: Incluímos o erro detalhado do subprocesso
        detailed_error = proc.stderr or proc.stdout or "rvc script failed without error message."
        raise RuntimeError(detailed_error)


_is_downloading = False


def _run_download_script():
    global _is_downloading
    try:
        subprocess.run([RVC_PYTHON, "download_popular_models.py"])
    except Exception as e:
        print(f"[error] Failed to download popular models: {e}", file=sys.stderr)
    finally:
        _is_downloading = False


@app.get("/voices")
def voices():
    """Lists voices (RVC models) ready to sing."""
    # Adicionamos uma verificação extra para ter certeza
    print("Listing voices. Found folders:", os.listdir(RVC_MODELS_DIR) if os.path.exists(RVC_MODELS_DIR) else "Not Found")
    return {
        "voices": _list_voices(),
        "ready": os.path.exists(RVC_SCRIPT),
        "downloading": _is_downloading
    }


@app.post("/voices/download")
async def download_voices():
    global _is_downloading
    if _is_downloading:
        return {"status": "downloading", "msg": "Download already in progress"}
    
    _is_downloading = True
    asyncio.create_task(asyncio.to_thread(_run_download_script))
    return {"status": "started", "msg": "Download started in background"}


@app.post("/convert")
async def convert(request: Request):
    if not os.path.exists(RVC_SCRIPT):
        raise HTTPException(503, "RVC environment not installed.")
    model = request.headers.get("x-model", "").strip()
    if not model:
        raise HTTPException(400, "X-Model header missing.")
    try:
        pitch = int(request.headers.get("x-pitch", "0"))
    except ValueError:
        pitch = 0

    data = await request.body()
    if not data:
        raise HTTPException(400, "Audio file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Audio file is too large.")

    work = tempfile.mkdtemp(prefix="samplebox_vc_")
    src = os.path.join(work, "src.wav")
    with open(src, "wb") as f:
        f.write(data)

    dst = os.path.join(work, "dst.wav")
    device = "cpu"
    
    async with _get_lock():
        try:
            await asyncio.to_thread(_run_rvc, model, src, dst, pitch, device)
        except RuntimeError as e:
            shutil.rmtree(work, ignore_errors=True)
            # AQUI ESTÁ A MUDANÇA: Usamos o erro detalhado na resposta
            raise HTTPException(500, f"Voice conversion failed. Details: {e}")

    if not os.path.exists(dst):
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, "Vocal not generated.")

    cleanup = BackgroundTask(shutil.rmtree, work, ignore_errors=True)
    return FileResponse(
        dst,
        media_type="audio/wav",
        filename="vocal_converted.wav",
        background=cleanup,
    )
