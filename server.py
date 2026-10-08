"""
Sample Box — AI Voice Engine (RVC) — Production @ Render GPU
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import torch  # Import torch

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

app = FastAPI(title="Sample Box — Vocal Extractor & RVC")

# Configure CORS
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
_origins = ["*"] if _origins_env in ("", "*") else [o.strip() for o in _origins_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
MAX_BYTES = 50 * 1024 * 1024  # 50 MB
RVC_PYTHON = sys.executable
RVC_SCRIPT = "rvc_infer.py"
RVC_MODELS_DIR = "models"
_lock = asyncio.Lock()


# --- Health & Voice Listing ---
@app.get("/health")
def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    return {
        "ok": True,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
    }

def _list_voices() -> "list[str]":
    if not os.path.isdir(RVC_MODELS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RVC_MODELS_DIR)):
        folder = os.path.join(RVC_MODELS_DIR, name)
        if os.path.isdir(folder) and any(f.endswith(".pth") for f in os.listdir(folder)):
            out.append(name)
    return out

@app.get("/voices")
def voices():
    return {
        "voices": _list_voices(),
        "ready": os.path.exists(RVC_SCRIPT),
    }

# --- RVC Inference ---
def _run_rvc(model: str, src: str, dst: str, pitch: int, device: str) -> None:
    cmd = [
        RVC_PYTHON, RVC_SCRIPT,
        "--model", model,
        "--input", src,
        "--output", dst,
        "--pitch", str(pitch),
        "--device", device,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detailed_error = proc.stderr or proc.stdout or "RVC script failed without error message."
        raise RuntimeError(detailed_error)

@app.post("/convert")
async def convert(request: Request):
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
    
    # AQUI ESTÁ A MÁGICA: Detecta a GPU automaticamente!
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"--> Using device: {device}")
    
    async with _lock:
        try:
            await asyncio.to_thread(_run_rvc, model, src, dst, pitch, device)
        except RuntimeError as e:
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(500, f"Voice conversion failed. Details: {e}")

    if not os.path.exists(dst):
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, "Converted audio not generated.")

    cleanup = BackgroundTask(shutil.rmtree, work, ignore_errors=True)
    return FileResponse(dst, media_type="audio/wav", filename="vocal_converted.wav", background=cleanup)

