"""
RunPod Serverless Handler for CTT Vision Service.
Uses runpod_model_loader (direct HuggingFace transformers, no Ollama).

Usage (RunPod):
    Docker: see Dockerfile.runpod
    Entrypoint: python -u runpod_handler.py

Author: Closing Time Technologies LLC
Created: 2026-03-27
"""

import json
import time

import runpod
import traceback
import os

print("[RunPod] Container starting...")
print(f"[RunPod] Python: {__import__('sys').version}")
print(f"[RunPod] HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
print(f"[RunPod] /runpod-volume exists: {os.path.exists('/runpod-volume')}")
if os.path.exists('/runpod-volume'):
    print(f"[RunPod] /runpod-volume contents: {os.listdir('/runpod-volume')}")
    if os.path.exists('/runpod-volume/hf_cache'):
        print(f"[RunPod] hf_cache contents: {os.listdir('/runpod-volume/hf_cache')}")

try:
    import torch
    print(f"[RunPod] PyTorch: {torch.__version__}")
    print(f"[RunPod] CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[RunPod] GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[RunPod] PyTorch check failed: {e}")

try:
    from runpod_model_loader import generate, check_health, parse_json_response, load_model
    print("[RunPod] Loading model at startup...")
    load_model()
    print("[RunPod] Model loaded. Handler ready.")
except Exception as e:
    _startup_error = str(e)
    print(f"[RunPod] MODEL LOAD FAILED: {_startup_error}")
    traceback.print_exc()
    generate = None
    check_health = lambda: {"error": _startup_error, "model_loaded": False}
    parse_json_response = None
    load_model = None


def handler(event):
    start = time.time()
    input_data = event.get("input", {})
    action = input_data.get("action")

    if action == "health":
        return {"ok": True, "data": check_health()}

    if generate is None:
        return {"ok": False, "error": "Model not loaded", "code": "MODEL_NOT_LOADED"}

    image = input_data.get("image")  # base64
    if not image:
        return {"ok": False, "error": "Missing 'image' field", "code": "MISSING_IMAGE"}

    if action == "ocr":
        prompt = "Extract all text from this image. Return the raw text exactly as it appears."
        result = generate(image, prompt)
        if result.get("error"):
            return {"ok": False, "error": result["error"], "code": "MODEL_ERROR"}
        return {
            "ok": True,
            "data": {
                "text": result["text"],
                "processing_ms": result["processing_ms"],
            },
        }

    elif action == "extract":
        schema_fields = input_data.get("fields", [])
        doc_type = input_data.get("document_type", "document")
        fields_str = ", ".join(schema_fields)
        prompt = (
            f"You are a document extraction specialist analyzing a {doc_type}. "
            f"Extract these fields from the document: {fields_str}. "
            "Return JSON only with field names as keys. For each field, return an object "
            'with "value" (extracted value or null if not found) and "confidence" (0.0-1.0). '
            "Return ONLY valid JSON, no explanation."
        )
        result = generate(image, prompt, temperature=0.0)
        if result.get("error"):
            return {"ok": False, "error": result["error"], "code": "MODEL_ERROR"}
        parsed = parse_json_response(result["text"])
        avg_confidence = None
        if parsed:
            confidences = [
                v.get("confidence", 0) for v in parsed.values()
                if isinstance(v, dict) and "confidence" in v
            ]
            if confidences:
                avg_confidence = round(sum(confidences) / len(confidences), 3)
        return {
            "ok": True,
            "data": {
                "fields": parsed or {},
                "raw_response": result["text"] if not parsed else None,
                "confidence": avg_confidence,
                "processing_ms": result["processing_ms"],
            },
        }

    elif action == "classify":
        categories = input_data.get("categories", [])
        categories_str = ", ".join(categories)
        prompt = (
            f"Classify this document into one of these categories: {categories_str}. "
            "Return JSON with: "
            '"classification" (best match from the list), '
            '"confidence" (0.0-1.0), '
            '"alternate" (dict of other likely categories with their confidence scores). '
            "Return ONLY valid JSON, no explanation."
        )
        result = generate(image, prompt, temperature=0.0)
        if result.get("error"):
            return {"ok": False, "error": result["error"], "code": "MODEL_ERROR"}
        parsed = parse_json_response(result["text"])
        classification = None
        confidence = None
        alternate = {}
        if parsed:
            classification = parsed.get("classification")
            confidence = parsed.get("confidence")
            alternate = parsed.get("alternate", {})
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

    elif action == "describe":
        prompt = input_data.get("prompt", "Describe this image.")
        result = generate(image, prompt)
        if result.get("error"):
            return {"ok": False, "error": result["error"], "code": "MODEL_ERROR"}
        parsed = parse_json_response(result["text"])
        return {
            "ok": True,
            "data": {
                "response": result["text"],
                "structured": parsed,
                "processing_ms": result["processing_ms"],
            },
        }

    return {"ok": False, "error": f"Unknown action: {action}", "code": "UNKNOWN_ACTION"}


runpod.serverless.start({"handler": handler})
