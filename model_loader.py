"""
Qwen3-VL model loader via Ollama.
Handles health checks, generation, and response parsing.

Author: Closing Time Technologies LLC
Created: 2026-03-27
"""

import base64
import json
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:8b")


def check_ollama() -> dict:
    """Check if Ollama is running and the vision model is available."""
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        model_loaded = any(VISION_MODEL in m for m in models)
        return {
            "ollama_running": True,
            "model": VISION_MODEL,
            "model_available": model_loaded,
            "available_models": models,
        }
    except Exception as e:
        return {
            "ollama_running": False,
            "model": VISION_MODEL,
            "model_available": False,
            "error": str(e),
        }


def generate(image_b64: str, prompt: str, temperature: float = 0.0) -> dict:
    """
    Send an image + prompt to Qwen3-VL via Ollama and return the response.

    Args:
        image_b64: Base64-encoded image data (no data URI prefix).
        prompt: The text prompt to send alongside the image.
        temperature: LLM temperature (0.0 for extraction/classification).

    Returns:
        dict with 'text', 'processing_ms', and 'raw_response'.
    """
    start = time.time()

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()

        processing_ms = round((time.time() - start) * 1000)
        response_text = data.get("response", "")

        return {
            "text": response_text,
            "processing_ms": processing_ms,
            "done": data.get("done", False),
        }

    except httpx.TimeoutException:
        processing_ms = round((time.time() - start) * 1000)
        return {
            "text": "",
            "processing_ms": processing_ms,
            "error": "Ollama request timed out (120s)",
        }
    except Exception as e:
        processing_ms = round((time.time() - start) * 1000)
        logger.error(f"[ModelLoader] Generate failed: {e}")
        return {
            "text": "",
            "processing_ms": processing_ms,
            "error": str(e),
        }


def parse_json_response(text: str) -> dict | None:
    """Attempt to parse JSON from the model response, handling markdown fences."""
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None
