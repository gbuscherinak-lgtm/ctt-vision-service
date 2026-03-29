"""
Direct model loader for RunPod serverless — no Ollama dependency.
Loads Qwen2.5-VL-7B-Instruct via HuggingFace transformers.

This is the RunPod path. Local development uses model_loader.py (Ollama).

Model 1 of 3 in the CTT Vision stack:
  - Qwen2.5-VL-7B: Classification, general vision, business cards
  - PaddleOCR-VL 7B: Document parsing, table extraction (future)
  - DeepSeek-OCR 3B: Receipt parsing, fast OCR (future)

Author: Closing Time Technologies LLC
Created: 2026-03-27
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    if "decord" in str(e):
        # decord not needed for image-only inference — stub it
        import sys
        sys.modules['decord'] = type(sys)('decord')
        sys.modules['decord'].VideoReader = None
        sys.modules['decord'].cpu = lambda: None
        from qwen_vl_utils import process_vision_info
    else:
        raise

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_model = None
_processor = None


def load_model():
    """Load model and processor. Called once at container start."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    print(f"[RunPodLoader] Loading {MODEL_ID}...")
    start = time.time()

    try:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _processor = AutoProcessor.from_pretrained(MODEL_ID)

        elapsed = round(time.time() - start, 1)
        print(f"[RunPodLoader] Model loaded in {elapsed}s on {_model.device}")
        return _model, _processor
    except Exception as e:
        print(f"[RunPodLoader] FATAL: Model load failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def check_health() -> dict:
    """Return model status for health checks."""
    return {
        "model": MODEL_ID,
        "model_loaded": _model is not None,
        "runtime": "runpod_transformers",
        "device": str(_model.device) if _model else "not loaded",
    }


def generate(image_base64: str, prompt: str, temperature: float = 0.0) -> dict:
    """
    Send an image + prompt to Qwen2.5-VL and return the response.

    Args:
        image_base64: Base64-encoded image data.
        prompt: Text prompt to send alongside the image.
        temperature: LLM temperature (0.0 for extraction/classification).

    Returns:
        dict with 'text' and 'processing_ms', or 'error' on failure.
    """
    start = time.time()

    try:
        model, processor = load_model()

        # Decode image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Build messages in Qwen VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process with Qwen2.5-VL pipeline
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate
        gen_kwargs = {"max_new_tokens": 2048}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        generated_ids = model.generate(**inputs, **gen_kwargs)

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        processing_ms = round((time.time() - start) * 1000)
        return {"text": output_text, "processing_ms": processing_ms}

    except Exception as e:
        processing_ms = round((time.time() - start) * 1000)
        logger.error(f"[RunPodLoader] Generate failed: {e}")
        return {"text": "", "processing_ms": processing_ms, "error": str(e)}


def parse_json_response(text: str) -> dict | None:
    """Attempt to parse JSON from model response, handling markdown fences."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None
