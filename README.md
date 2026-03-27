# CTT Vision Service

Unified AI vision microservice for all Closing Time Technologies products.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy env and configure
cp .env.example .env

# Pull the vision model
ollama pull qwen3-vl:8b

# Start the service
python3 server.py
# or: uvicorn server:app --port 8790 --reload
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/v1/health | Status, model info, uptime |
| POST | /api/v1/ocr | Raw text extraction from image |
| POST | /api/v1/extract | Structured field extraction with schema |
| POST | /api/v1/classify | Document classification into categories |
| POST | /api/v1/describe | Freeform image understanding |

## Auth

Set `CTT_VISION_KEY` in `.env`. Pass as `X-CTT-Vision-Key` header. Skipped if env var is empty (local dev).

## Products

- **Theo** (port 8787) — Real estate transaction management
- **REOM** (port 5050) — Real estate office manager
- **TipSplit** — Receipt parsing (planned)
