# CLAUDE.md — CTT Vision Service

## Project Purpose
CTT Vision Service — unified AI vision microservice for all Closing Time Technologies products (Theo, REOM, TipSplit, future products). Wraps Qwen3-VL via Ollama for OCR, document extraction, classification, and image understanding.

## Tech Stack
- **Backend:** Python 3.11+ / FastAPI
- **AI:** Qwen3-VL (8b) via Ollama (Apache 2.0 — zero licensing risk)
- **Port:** 8790
- **Auth:** X-CTT-Vision-Key header (env var)

## How to Run
```bash
uvicorn server:app --port 8790 --reload
# or: python3 server.py
```

## Endpoint Reference

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/v1/health | Model status, uptime, request count |
| POST | /api/v1/ocr | Raw OCR text extraction |
| POST | /api/v1/extract | Schema-driven field extraction (temp 0.0) |
| POST | /api/v1/classify | Document classification (temp 0.0) |
| POST | /api/v1/describe | Freeform vision prompt |

## API Response Format
```json
{"ok": true, "data": {...}}
{"ok": false, "error": "msg", "code": "CODE"}
```

## Coding Standards
- Snake_case for all API JSON keys
- {"ok": true/false} response wrapper on every endpoint
- Temperature 0.0 for extraction and classification
- Every request logged to logs/vision_service_{date}.jsonl
- No hardcoded API keys — use .env

## Key Files
- `server.py` — FastAPI app, all endpoints
- `model_loader.py` — Ollama integration, generate(), JSON parsing
- `schemas/` — Per-product extraction schemas (JSON)
- `logs/` — JSONL request logs (gitignored)

## Products Using This Service
- **Theo** — Real estate TC platform (port 8787)
- **REOM** — Office manager for brokers (port 5050)
- **TipSplit** — Receipt parsing (planned)

## Deployment
- Phase 1: Local only (this phase)
- Phase 2: RunPod GPU deployment (Dockerfile ready)
- Phase 3: Production integration with Theo + REOM
