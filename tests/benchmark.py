"""
Benchmark script for CTT Vision Service.
Tests all endpoints with a sample image.

Usage: python3 tests/benchmark.py
"""

import base64
import json
import sys
import time

import httpx

BASE_URL = "http://localhost:8790"


def create_test_image_b64():
    """Create a minimal 1x1 white PNG as base64 for health-check-level testing."""
    # Minimal valid PNG (1x1 white pixel)
    import struct
    import zlib

    def png_chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_data = b"\x00\xFF\xFF\xFF"
    idat = png_chunk(b"IDAT", zlib.compress(raw_data))
    iend = png_chunk(b"IEND", b"")

    png_bytes = signature + ihdr + idat + iend
    return base64.b64encode(png_bytes).decode()


def test_health():
    print("=== /api/v1/health ===")
    resp = httpx.get(f"{BASE_URL}/api/v1/health", timeout=10.0)
    data = resp.json()
    print(json.dumps(data, indent=2))
    return data.get("ok", False)


def test_ocr(image_b64):
    print("\n=== /api/v1/ocr ===")
    start = time.time()
    resp = httpx.post(
        f"{BASE_URL}/api/v1/ocr",
        json={"image": image_b64, "mime_type": "image/png", "product": "benchmark"},
        timeout=120.0,
    )
    elapsed = round((time.time() - start) * 1000)
    data = resp.json()
    print(f"Status: {resp.status_code} | Round-trip: {elapsed}ms")
    print(json.dumps(data, indent=2))
    return data.get("ok", False)


def test_classify(image_b64):
    print("\n=== /api/v1/classify ===")
    start = time.time()
    resp = httpx.post(
        f"{BASE_URL}/api/v1/classify",
        json={
            "image": image_b64,
            "mime_type": "image/png",
            "categories": ["RPA", "SCO", "BCO", "ADM", "OTHER"],
            "product": "benchmark",
        },
        timeout=120.0,
    )
    elapsed = round((time.time() - start) * 1000)
    data = resp.json()
    print(f"Status: {resp.status_code} | Round-trip: {elapsed}ms")
    print(json.dumps(data, indent=2))
    return data.get("ok", False)


def test_extract(image_b64):
    print("\n=== /api/v1/extract ===")
    start = time.time()
    resp = httpx.post(
        f"{BASE_URL}/api/v1/extract",
        json={
            "image": image_b64,
            "mime_type": "image/png",
            "schema_fields": ["purchase_price", "close_of_escrow_date", "emd_amount"],
            "document_type": "RPA",
            "product": "benchmark",
        },
        timeout=120.0,
    )
    elapsed = round((time.time() - start) * 1000)
    data = resp.json()
    print(f"Status: {resp.status_code} | Round-trip: {elapsed}ms")
    print(json.dumps(data, indent=2))
    return data.get("ok", False)


if __name__ == "__main__":
    print("CTT Vision Service Benchmark")
    print("=" * 40)

    image_b64 = create_test_image_b64()

    # If a real image file is provided as argument, use that instead
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        print(f"Using image: {sys.argv[1]}")
    else:
        print("Using synthetic 1x1 test image (pass a real image path for meaningful results)")

    results = {}
    results["health"] = test_health()
    results["ocr"] = test_ocr(image_b64)
    results["classify"] = test_classify(image_b64)
    results["extract"] = test_extract(image_b64)

    print("\n" + "=" * 40)
    print("RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
