# CLAUDE.md — CTT Vision Service

## Project Purpose
CTT Vision Service — unified AI vision microservice for all Closing Time Technologies products (Theo, REOM, TipSplit, future products). Multi-model routing behind a single API.

## Three-Model Stack (Locked 2026-03-27)

| Model | Size | Task | Products |
|-------|------|------|----------|
| Qwen2.5-VL-7B | 7B | Classification, general vision, business cards | Theo classifier, REOM |
| PaddleOCR-VL 7B | 7B | Document parsing, table extraction, form reading | Theo, REOM attachments |
| DeepSeek-OCR 3B | 3B | Receipt parsing, fast OCR, structured data | TipSplit, quick OCR |

All Apache 2.0. All run on the same GPU. Products don't know or care which model handles their request.

**Current status:** Qwen2.5-VL-7B deployed. PaddleOCR-VL and DeepSeek-OCR queued.

## Two Runtime Paths

| Path | Files | Backend | Use |
|------|-------|---------|-----|
| Local | server.py + model_loader.py | Ollama (qwen3-vl:8b) | Dev/testing |
| RunPod | runpod_handler.py + runpod_model_loader.py | HuggingFace transformers (GPU) | Production |

## Tech Stack
- **Local:** Python 3.11+ / FastAPI / Ollama
- **RunPod:** PyTorch + transformers + CUDA (Dockerfile.runpod)
- **Port:** 8790 (local)
- **Auth:** X-CTT-Vision-Key header (env var)

## How to Run (Local)
```bash
uvicorn server:app --port 8790 --reload
# or: python3 server.py
```

## Endpoint Reference

| Method | Path | Purpose | Model (RunPod) |
|--------|------|---------|----------------|
| GET | /api/v1/health | Model status, uptime | — |
| POST | /api/v1/ocr | Raw OCR text extraction | DeepSeek-OCR (future) |
| POST | /api/v1/extract | Schema-driven field extraction | PaddleOCR-VL (future) |
| POST | /api/v1/classify | Document classification | Qwen2.5-VL |
| POST | /api/v1/describe | Freeform vision prompt | Qwen2.5-VL |

## API Response Format
```json
{"ok": true, "data": {...}}
{"ok": false, "error": "msg", "code": "CODE"}
```

## Key Files
- `server.py` — FastAPI app, all endpoints (local path)
- `model_loader.py` — Ollama integration (local path)
- `runpod_handler.py` — RunPod serverless handler (production path)
- `runpod_model_loader.py` — Direct HuggingFace model loading (production path)
- `schemas/` — Per-product extraction schemas (JSON)
- `logs/` — JSONL request logs (gitignored, local only)

## Coding Standards
- Snake_case for all API JSON keys
- {"ok": true/false} response wrapper on every endpoint
- Temperature 0.0 for extraction and classification
- No hardcoded API keys — use .env
- No Ollama in RunPod containers — direct transformers only

## Products
- **Theo** — Real estate TC platform (port 8787)
- **REOM** — Office manager for brokers (port 5050)
- **TipSplit** — Receipt parsing (mobile app)
