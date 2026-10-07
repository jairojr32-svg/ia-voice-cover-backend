"""
Conversão de voz cantada (RVC) — chamado por subprocess pelo backend FastAPI.

Recebe um vocal isolado e o re-canta com a voz de um modelo .pth treinado,
PRESERVANDO a melodia/tom/ritmo originais (singing voice conversion).

Uso:
  python rvc_infer.py --model NOME --input in.wav --output out.wav [--pitch 0] [--device mps:0]

Os modelos ficam em backend/rvc/models/<NOME>/<NOME>.pth (+ <NOME>.index opcional).
"""

import os
import sys

# Prevent Intel OpenMP duplicate library initialization segmentation faults on macOS (extremely common with PyTorch + FAISS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import glob
import warnings
import functools

warnings.filterwarnings("ignore")

# Monkeypatch torch.load to bypass PyTorch >= 2.6 weights_only=True restriction for older/custom RVC & fairseq checkpoints
import torch
_orig_load = torch.load

@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)

torch.load = _patched_load

# Force PyTorch to report MPS as unavailable to bypass MPS-specific segmentation faults on macOS (RMVPE crash)
torch.backends.mps.is_available = lambda: False

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")


def _resolve_model(name: str):
    """Retorna (pth, index) do modelo NOME na pasta models/."""
    folder = os.path.join(MODELS_DIR, name)
    if not os.path.isdir(folder):
        sys.exit(f"[rvc] modelo '{name}' não encontrado em {folder}")
    pths = glob.glob(os.path.join(folder, "*.pth"))
    if not pths:
        sys.exit(f"[rvc] nenhum .pth dentro de {folder}")
    indexes = glob.glob(os.path.join(folder, "*.index"))
    return pths[0], (indexes[0] if indexes else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="nome da pasta do modelo em models/")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pitch", type=int, default=0, help="semitons (0 = mantém o tom)")
    ap.add_argument("--device", default="mps:0")
    ap.add_argument("--f0method", default="rmvpe")
    ap.add_argument("--index-rate", type=float, default=0.5)
    args = ap.parse_args()

    from rvc_python.infer import RVCInference

    pth, index = _resolve_model(args.model)

    rvc = RVCInference(device=args.device)
    try:
        print(f"Loading {args.model} as RVC v2...")
        rvc.load_model(pth, version="v2", index_path=index)
    except Exception as e:
        if "size mismatch" in str(e) or "256" in str(e) or "768" in str(e) or "Synthesizer" in str(e):
            print(f"Size mismatch detected. Falling back to RVC v1 for {args.model}...")
            rvc = RVCInference(device=args.device)
            rvc.load_model(pth, version="v1", index_path=index)
        else:
            raise e

    rvc.set_params(
        f0method=args.f0method,
        f0up_key=args.pitch,
        index_rate=args.index_rate if index else 0.0,
    )
    rvc.infer_file(args.input, args.output)

    if not os.path.exists(args.output):
        sys.exit("[rvc] saída não gerada")
    print(f"[rvc] ok -> {args.output}")


if __name__ == "__main__":
    main()
