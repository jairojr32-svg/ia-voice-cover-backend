"""
Conversão de voz cantada (RVC) — chamado por subprocess pelo backend FastAPI.
VERSÃO DE DEPURAÇÃO ROBUSTA
"""

import os
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import glob
import warnings

warnings.filterwarnings("ignore")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")


def _resolve_model(name: str):
    """Retorna o caminho do .pth e desativa o .index para estabilidade."""
    logging.info(f"Resolving model for '{name}'...")
    folder = os.path.join(MODELS_DIR, name)
    if not os.path.isdir(folder):
        logging.error(f"Model folder not found: {folder}")
        sys.exit(f"Modelo '{name}' não encontrado.")
    
    pths = glob.glob(os.path.join(folder, "*.pth"))
    if not pths:
        logging.error(f"No .pth file found in {folder}")
        sys.exit(f"Nenhum arquivo .pth dentro de {folder}.")
    
    logging.info(f"Found model: {pths[0]}. Index file DISABLED for stability.")
    return pths[0], ""


def main():
    logging.info("--- RVC Inference Script Started ---")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--f0method", default="rmvpe")
    ap.add_argument("--index-rate", type=float, default=0.0)
    args = ap.parse_args()

    logging.info(f"Arguments: {args}")
    args.index_rate = 0.0 # Força a desativação do index

    from rvc_python.infer import RVCInference
    logging.info("RVCInference imported successfully.")

    pth, index = _resolve_model(args.model)

    effective_device = args.device
    if "cuda" in effective_device and not torch.cuda.is_available():
        logging.warning("CUDA specified but not available. Falling back to CPU.")
        effective_device = "cpu"

    rvc = RVCInference(device=effective_device)
    logging.info(f"RVCInference initialized on device: {effective_device}")
    
    try:
        logging.info(f"Attempting to load model {args.model}...")
        rvc.load_model(pth, version="v2", index_path=index)
    except Exception as e:
        logging.warning(f"Failed to load as v2, trying v1. Error: {e}")
        try:
            rvc.load_model(pth, version="v1", index_path=index)
        except Exception as e2:
            logging.error("Failed to load model as v1 or v2.", exc_info=True)
            raise e2
    
    logging.info("Model loaded successfully.")

    rvc.set_params(f0method=args.f0method, f0up_key=args.pitch, index_rate=args.index_rate)
    logging.info("Parameters set. Starting inference...")
    rvc.infer_file(args.input, args.output)
    logging.info("Inference completed.")

    if not os.path.exists(args.output):
        logging.error("Output file was not generated.")
        sys.exit("Saída não gerada.")
    
    logging.info(f"--- RVC Inference Script Finished Successfully -> {args.output} ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An unhandled exception occurred in main()", exc_info=True)
        sys.exit(1)
