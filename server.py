"""
CTT Vision Service — unified AI vision for all CTT products.
FastAPI + Ollama + Qwen3-VL.

Author: Closing Time Technologies LLC
Created: 2026-03-27
Port: 8790
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import model_loader

load_dotenv()

# ============================================
# LOGGING
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ctt-vision")

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def log_request(endpoint: str, product: str, processing_ms: int, confidence: float = None):
    """Append a JSONL log entry for every request."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "product": product,
        "processing_ms": processing_ms,
        "confidence": confidence,
    }
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"vision_service_{date_str}.jsonl")
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write log: {e}")


# ============================================
# APP SETUP
# ============================================

app = FastAPI(title="CTT Vision Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8787",   # Theo
        "http://localhost:5050",   # REOM
        "http://localhost:3000",   # Dev frontends
        "http://127.0.0.1:8787",
        "http://127.0.0.1:5050",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
REQUEST_COUNT = 0
CTT_VISION_KEY = os.getenv("CTT_VISION_KEY", "")


# ============================================
# AUTH
# ============================================

def verify_key(x_ctt_vision_key: str = Header(None)):
    """Validate API key. Skip if no key is configured (local dev)."""
    if CTT_VISION_KEY and x_ctt_vision_key != CTT_VISION_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-CTT-Vision-Key")


# ============================================
# REQUEST MODELS
# ============================================

class OCRRequest(BaseModel):
    image: str  # base64
    mime_type: str = "image/png"
    product: str = "unknown"

class ExtractRequest(BaseModel):
    image: str  # base64
    mime_type: str = "image/png"
    schema_fields: list[str]
    document_type: str = "document"
    product: str = "unknown"

class ClassifyRequest(BaseModel):
    image: str  # base64
    mime_type: str = "image/png"
    categories: list[str]
    product: str = "unknown"

class DescribeRequest(BaseModel):
    image: str  # base64
    mime_type: str = "image/png"
    prompt: str
    product: str = "unknown"


# ============================================
# ENDPOINTS
# ============================================

@app.get("/api/v1/health")
def health():
    """Health check — model status, uptime, request count."""
    status = model_loader.check_ollama()
    return {
        "ok": True,
        "data": {
            **status,
            "uptime_seconds": round(time.time() - START_TIME),
            "request_count": REQUEST_COUNT,
        },
    }


@app.post("/api/v1/ocr")
def ocr(req: OCRRequest, x_ctt_vision_key: str = Header(None)):
    """Extract all text from an image."""
    global REQUEST_COUNT
    verify_key(x_ctt_vision_key)
    REQUEST_COUNT += 1

    prompt = "Extract all text from this image. Return the raw text exactly as it appears."

    result = model_loader.generate(req.image, prompt)

    if result.get("error"):
        return {"ok": False, "error": result["error"], "code": "MODEL_ERROR"}

    log_request("/ocr", req.product, result["processing_ms"])

    return {
        "ok": True,
        "data": {
            "text": result["text"],
            "processing_ms": result["processing_ms"],
        },
    }


@app.post("/api/v1/extract")
def extract(req: ExtractRequest, x_ctt_vision_key: str = Header(None)):
    """Extract structured fields from a document image."""
    global REQUEST_COUNT
    verify_key(x_ctt_vision_key)
    REQUEST_COUNT += 1

    fields_str = ", ".join(req.schema_fields)
    prompt = (
        f"You are a document extraction specialist analyzing a {req.document_type}. "
        f"Extract these fields from the document: {fields_str}. "
        "Return JSON only with field names as keys. For each field, return an object "
        'with "value" (extracted value or null if not found) and "confidence" (0.0-1.0). '
        "Example: {\"field_name\": {\"value\": \"...\", \"confidence\": 0.95}}. "
        "Return ONLY valid JSON, no explanation."
    )

    result = model_loader.generate(req.image, prompt, temperature=0.0)

    if result.get("error"):
        return {"ok": False, "error": result["error"], "code": "MODEL_ERROR"}

    parsed = model_loader.parse_json_response(result["text"])

    # Calculate average confidence
    avg_confidence = None
    if parsed:
        confidences = [
            v.get("confidence", 0) for v in parsed.values()
            if isinstance(v, dict) and "confidence" in v
        ]
        if confidences:
            avg_confidence = round(sum(confidences) / len(confidences), 3)

    log_request("/extract", req.product, result["processing_ms"], avg_confidence)

    return {
        "ok": True,
        "data": {
            "fields": parsed or {},
            "raw_response": result["text"] if not parsed else None,
            "confidence": avg_confidence,
            "processing_ms": result["processing_ms"],
        },
    }


@app.post("/api/v1/classify")
def classify(req: ClassifyRequest, x_ctt_vision_key: str = Header(None)):
    """Classify a document image into one of the given categories."""
    global REQUEST_COUNT
    verify_key(x_ctt_vision_key)
    REQUEST_COUNT += 1

    categories_str = ", ".join(req.categories)
    prompt = (
        f"Classify this document into one of these categories: {categories_str}. "
        "Return JSON with: "
        '"classification" (best match from the list), '
        '"confidence" (0.0-1.0), '
        '"alternate" (dict of other likely categories with their confidence scores). '
        "Return ONLY valid JSON, no explanation."
    )

    result = model_loader.generate(req.image, prompt, temperature=0.0)

    if result.get("error"):
        return {"ok": False, "error": result["error"], "code": "MODEL_ERROR"}

    parsed = model_loader.parse_json_response(result["text"])

    classification = None
    confidence = None
    alternate = {}

    if parsed:
        classification = parsed.get("classification")
        confidence = parsed.get("confidence")
        alternate = parsed.get("alternate", {})

    log_request("/classify", req.product, result["processing_ms"], confidence)

    return {
        "ok": True,
        "data": {
            "classification": classification,
            "confidence": confidence,
            "alternate": alternate,
            "raw_response": result["text"] if not parsed else None,
            "processing_ms": result["processing_ms"],
        },
    }


@app.post("/api/v1/describe")
def describe(req: DescribeRequest, x_ctt_vision_key: str = Header(None)):
    """Freeform image understanding — user provides the prompt."""
    global REQUEST_COUNT
    verify_key(x_ctt_vision_key)
    REQUEST_COUNT += 1

    result = model_loader.generate(req.image, req.prompt)

    if result.get("error"):
        return {"ok": False, "error": result["error"], "code": "MODEL_ERROR"}

    parsed = model_loader.parse_json_response(result["text"])

    log_request("/describe", req.product, result["processing_ms"])

    return {
        "ok": True,
        "data": {
            "response": result["text"],
            "structured": parsed,
            "processing_ms": result["processing_ms"],
        },
    }


# ============================================
# STARTUP
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8790"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
