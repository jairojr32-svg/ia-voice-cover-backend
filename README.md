---
title: Sample Box Vocal Extractor
emoji: 🎧
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Sample Box — Vocal Extractor (motor Demucs)

Motor de separação de vocal (Demucs `htdemucs`) + detecção de BPM/tom (librosa),
empacotado como container Docker para rodar **grátis** no Hugging Face Spaces (CPU).

O front-end (Next.js na Vercel) chama este Space via `NEXT_PUBLIC_API_URL`.

## Endpoints
- `GET /health` — status.
- `POST /analyze` — corpo = bytes do áudio, header `x-filename`. Retorna `{ bpm, key, label_pt, ... }`.
- `POST /separate` — corpo = bytes do áudio, header `x-filename`. Retorna o `vocal.wav` isolado.
  Header opcional `x-preview-seconds` para processar só os primeiros N segundos.

## Configuração
- `ALLOWED_ORIGINS` — domínios liberados no CORS (separados por vírgula). Ex.:
  `https://seu-site.vercel.app`. Padrão `*`.

## Notas
- Primeira separação baixa o modelo `htdemucs` (~algumas centenas de MB) e fica em cache.
- CPU grátis: uma música leva ~1–3 min. Para acelerar, o front pode mandar `x-preview-seconds`.
- Limite de 50 MB por arquivo.
