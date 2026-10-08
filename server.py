
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import torch
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

app = FastAPI(title="Sample Box - AI Engine [DEBUG MODE]")

_origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"])

RVC_PYTHON = sys.executable
RVC_SCRIPT = "rvc_infer.py"
RVC_MODELS_DIR = "models"
_lock = asyncio.Lock()

@app.get("/health")
def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    return {"ok": True, "gpu_available": gpu_available, "gpu_name": gpu_name}

def _list_voices():
    if not os.path.isdir(RVC_MODELS_DIR): return []
    return sorted([
        name for name in os.listdir(RVC_MODELS_DIR)
        if os.path.isdir(os.path.join(RVC_MODELS_DIR, name)) and any(f.endswith(".pth") for f in os.listdir(os.path.join(RVC_MODELS_DIR, name)))
    ])

@app.get("/voices")
def voices():
    return {"voices": _list_voices(), "ready": os.path.exists(RVC_SCRIPT)}

@app.get("/debug-info")
def debug_info():
    print("--- Running /debug-info endpoint ---")
    gpu_available = torch.cuda.is_available()
    
    model_files = {}
    if os.path.exists(RVC_MODELS_DIR):
        for root, _, files in os.walk(RVC_MODELS_DIR):
            for name in files:
                path = os.path.join(root, name)
                rel_path = os.path.relpath(path, RVC_MODELS_DIR)
                try:
                    size_mb = f"{os.path.getsize(path) / 1024 / 1024:.2f} MB"
                except OSError:
                    size_mb = "Size unreadable"
                model_files[rel_path] = size_mb
    else:
        model_files = {"error": "models directory not found"}

    test_execution_result = {}
    try:
        cmd = [RVC_PYTHON, RVC_SCRIPT, "--help"]
        print(f"Running test command: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        test_execution_result = {
            "status": "SUCCESS" if proc.returncode == 0 else "FAILED",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as e:
        test_execution_result = {"status": "CRASHED", "error": str(e), "traceback": traceback.format_exc()}

    return JSONResponse({
        "gpu_info": {"available": gpu_available, "name": torch.cuda.get_device_name(0) if gpu_available else "N/A"},
        "torch_version": torch.__version__,
        "cuda_version_from_torch": torch.version.cuda,
        "found_model_files": model_files,
        "inference_script_test_run": test_execution_result,
    })

def _run_rvc(model: str, src: str, dst: str, pitch: int, device: str):
    cmd = [RVC_PYTHON, RVC_SCRIPT, "--model", model, "--input", src, "--output", dst, "--pitch", str(pitch), "--device", device]
    print(f"Executing RVC command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detailed_error = proc.stderr or proc.stdout or "RVC script failed without error message."
        print(f"RVC SCRIPT FAILED:\n---STDOUT---\n{proc.stdout}\n---STDERR---\n{proc.stderr}\n---")
        raise RuntimeError(detailed_error)
    print("RVC script finished successfully.")

@app.post("/convert")
async def convert(request: Request):
    model = request.headers.get("x-model", "").strip()
    pitch = int(request.headers.get("x-pitch", "0"))
    data = await request.body()

    work = tempfile.mkdtemp(prefix="samplebox_vc_")
    src = os.path.join(work, "src.wav")
    with open(src, "wb") as f: f.write(data)
    dst = os.path.join(work, "dst.wav")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    try:
        await asyncio.to_thread(_run_rvc, model, src, dst, pitch, device)
        if not os.path.exists(dst):
            raise HTTPException(500, "Conversion ran but output file was not created.")
        
        cleanup = BackgroundTask(shutil.rmtree, work, ignore_errors=True)
        return FileResponse(dst, media_type="audio/wav", filename="vocal_converted.wav", background=cleanup)
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        print(f"FATAL ERROR during /convert: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Voice conversion failed. Details: {e}")

