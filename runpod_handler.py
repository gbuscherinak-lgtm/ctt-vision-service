"""
RunPod Serverless Handler for CTT Vision Service.
Wraps model_loader functions in RunPod's handler pattern.

Usage (RunPod):
    Set handler: runpod_handler.handler
    Docker: see Dockerfile.runpod

Author: Closing Time Technologies LLC
Created: 2026-03-27
"""

import json
import time

import runpod

from model_loader import generate, check_ollama, parse_json_response


def handler(event):
    start = time.time()
    input_data = event.get("input", {})
    action = input_data.get("action")

    if action == "health":
        return {"ok": True, "data": check_ollama()}

    image = input_data.get("image")  # base64
    if not image:
        return {"ok": False, "error": "Missing 'image' field", "code": "MISSING_IMAGE"}

    mime_type = input_data.get("mime_type", "image/png")

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
