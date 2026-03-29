# RunPod Serverless Deployment — Lessons Learned
**Date:** 2026-03-28 | **Model:** Qwen2.5-VL-7B-Instruct | **Service:** CTT Vision Service

---

## What We Were Deploying
- Qwen2.5-VL-7B-Instruct (16.6GB model)
- RunPod Serverless endpoint with queue-based workers
- RTX 4090 (24GB VRAM) target GPU

---

## Issues Encountered (in order)

### 1. Docker Base Image Tag Didn't Exist
**Error:** `failed to resolve source metadata for docker.io/runpod/pytorch:2.4.0-py3.11-cuda12.1.0-devel-ubuntu22.04: not found`

**Cause:** Made up a tag name that didn't exist on Docker Hub.

**Fix:** Check actual available tags. RunPod changed their naming convention to `1.0.3-cu1290-torch260-ubuntu2204` format.

**Lesson:** Always verify Docker image tags exist before using them. Use `curl` to query Docker Hub API or check the registry directly.

---

### 2. CUDA Version Too New for GPU Drivers
**Error:** `nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.9`

**Cause:** Used `runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204` — CUDA 12.9 requires newer NVIDIA drivers than what RTX 4090 workers have.

**Fix:** Use CUDA 12.1 base image: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`

**Lesson:** RTX 4090 workers on RunPod typically have driver ~550.x which supports up to CUDA 12.4. Don't use CUDA 12.9+ base images. Stick to CUDA 12.1 or 12.2 for maximum compatibility.

---

### 3. runpod/base Image — Container Create Exit Status 1
**Error:** `error creating container: container: create: container create: exit status 1`

**Cause:** `runpod/base:0.6.2-cuda12.2.0` has a different Python/entrypoint setup than expected.

**Fix:** Switched to official `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` which is battle-tested.

**Lesson:** Prefer official PyTorch Docker images over RunPod's custom base images for serverless. They're more widely tested and have predictable behavior.

---

### 4. Baked-in Model Made Image Too Large (24.5GB)
**Error:** Workers stuck in "throttled" state for 20+ minutes, image pull never completed.

**Cause:** Pre-downloading the 16.6GB model during Docker build created a 24.5GB image. RunPod's serverless image distribution couldn't deliver it to workers efficiently.

**Fix:** Removed model from image, download at runtime instead. Image dropped to ~3.5-6GB.

**Lesson:** Keep serverless Docker images under 10GB. Large images cause slow/failed image pulls on RunPod serverless. Use Network Volumes for large model files instead.

---

### 5. Image Pull "Pending" for Extended Periods
**Error:** Worker logs showed `image pull: pending` for 10-20 minutes with no download progress.

**Cause:** RunPod's container registry distribution is slow for first-time image pulls to certain data centers.

**Fix:** Patience. Eventually the pull starts. Subsequent pulls to the same data center are cached and fast.

**Lesson:** First deployment to a new region/data center will be slow. Plan for 10-20 minute cold starts on first deploy. This improves dramatically after the first pull.

---

### 6. PyTorch Version Too Old for Transformers
**Error:** `Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2`

**Cause:** Base image `pytorch/pytorch:2.2.2` has PyTorch 2.2.2, but `transformers>=4.45.0` requires PyTorch 2.4+ for Qwen2.5-VL model support.

**Fix:** Added `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121` before installing transformers.

**Lesson:** When using a PyTorch base image, check if your transformers version requires a newer PyTorch. Upgrade PyTorch in the Dockerfile before installing transformers.

---

### 7. `total_mem` vs `total_memory` Typo
**Error:** `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`

**Cause:** Diagnostic print used `total_mem` instead of `total_memory`.

**Fix:** Changed to `total_memory`.

**Lesson:** Small typo in diagnostic code can crash the entire container. Wrap all diagnostic prints in try/except.

---

### 8. Crash-Blind Debugging (No Error Visibility)
**Error:** Workers showed "worker exited with exit code 1" with no Python error output.

**Cause:** RunPod serverless doesn't capture stdout/stderr from containers that crash during startup. If `runpod_handler.py` crashes before calling `runpod.serverless.start()`, you get zero diagnostic info.

**Fix:** Wrapped the entire startup sequence in try/except. If model loading fails, the handler still registers with RunPod and returns the error in health check responses.

**Lesson:** ALWAYS wrap your RunPod handler startup in try/except. Never let an import or model load crash the process. Register the handler first, report errors through the API.

---

### 9. Init Timeout Too Short for Cold Start
**Error:** `context deadline exceeded` on container create.

**Cause:** RunPod's default init timeout (~7 min) is too short when image pull (5+ min) + model load (~18s) exceeds the window. Container gets killed before handler registers.

**Fix:** Set `RUNPOD_INIT_TIMEOUT=600` (10 min) in both Dockerfile ENV and RunPod Dashboard environment variables. Belt and suspenders.

**Lesson:** Always set `RUNPOD_INIT_TIMEOUT` explicitly for model-serving containers. Default is too aggressive for large images or slow first pulls.

---

### 10. decord Dependency Missing for qwen-vl-utils
**Error:** `ImportError: No module named 'decord'` at container startup.

**Cause:** `qwen-vl-utils` imports `decord` for video frame extraction. Not in requirements, crashes before handler registers.

**Fix:** Added `decord>=0.6.0` to requirements + `ffmpeg libsm6 libxext6` system deps in Dockerfile. Also added defensive import stub in `runpod_model_loader.py` as fallback (decord not needed for image-only inference).

**Lesson:** Check ALL transitive dependencies of vision libraries. `qwen-vl-utils` needs `decord` even if you only use image features.

---

### 11. Unpinned transformers Version
**Error:** Potential breaking changes in transformers 5.x for Qwen2.5-VL model class.

**Cause:** `transformers>=5.0` is an open-ended floor that lets pip resolve to untested versions.

**Fix:** Pinned to `transformers>=4.45.0,<5.0` — the range validated during the successful RTX 4090 pod test.

**Lesson:** Always pin upper bounds on ML framework dependencies. Breaking API changes between major versions are common.

---

## The Solution That Works

### Architecture
```
Docker Image (6GB):
  pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
  + pip install torch==2.4.1 (upgrade)
  + pip install transformers accelerate qwen-vl-utils runpod
  + runpod_handler.py + runpod_model_loader.py

Network Volume (ctt-vision-models, 20GB, US-NC-1):
  /hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/
  Pre-downloaded via Pod, persists forever

Serverless Endpoint:
  HF_HOME=/runpod-volume/hf_cache
  Workers mount Network Volume → instant model access
  No HuggingFace download at runtime
```

### Working Dockerfile
```dockerfile
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Upgrade PyTorch to 2.4+ (transformers requires it)
RUN pip install --no-cache-dir torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121

COPY requirements-runpod.txt .
RUN pip install --no-cache-dir -r requirements-runpod.txt

COPY runpod_handler.py .
COPY runpod_model_loader.py .
COPY schemas/ ./schemas/

ENV HF_HOME=/runpod-volume/hf_cache
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "runpod_handler.py"]
```

### Requirements (requirements-runpod.txt)
```
runpod>=1.7.0
transformers>=4.45.0
accelerate>=0.34.0
Pillow>=10.0.0
qwen-vl-utils>=0.0.8
```

**DO NOT include `torch` in requirements** — it's in the base image + Dockerfile upgrade.

---

## Network Volume Setup (One-Time)

1. Create Network Volume in RunPod Storage (20GB, same region as endpoint)
2. Deploy a temporary Pod with the volume attached
3. In Pod terminal: `HF_HOME=/workspace/hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')"`
4. Terminate Pod — model persists on volume
5. Attach volume to Serverless endpoint
6. Dockerfile uses `HF_HOME=/runpod-volume/hf_cache` to read from volume

---

## Do's and Don'ts

### DO
- Use official `pytorch/pytorch` base images
- Keep Docker images under 10GB
- Use Network Volumes for model weights
- Wrap handler startup in try/except
- Use CUDA 12.1 for maximum GPU driver compatibility
- Set idle timeout to 60+ seconds (model loading takes time)
- Verify Docker image tags exist before building
- Test on a Pod first before debugging Serverless

### DON'T
- Don't bake 15GB+ models into Docker images
- Don't use CUDA 12.9+ base images (driver incompatibility)
- Don't use `runpod/base` images for model serving (prefer pytorch/pytorch)
- Don't put `torch` in requirements.txt when using a PyTorch base image
- Don't let the handler crash before registering with RunPod
- Don't set idle timeout below 30 seconds
- Don't assume image pulls are fast on first deploy to a new region
- Don't guess Docker image tags — verify they exist

---

## Cost Notes
- Network Volume: ~$1.40/month (20GB × $0.07/GB)
- Serverless idle: $0/hr (min workers = 0)
- Serverless active: ~$0.00031/s for 24GB PRO GPU
- First cold start: ~5-10 min (image pull + model load)
- Warm requests: ~2-5 seconds
- Pod for model caching: ~$0.59/hr (terminate after download)

---

## GPU Compatibility Verified
- RTX 4090 (24GB VRAM) — CUDA 12.4 driver — WORKS
- L40S (48GB VRAM) — untested but should work
- Minimum VRAM: 16GB (model is ~14GB in bfloat16)
- Recommended: 24GB+

---

*Last updated: 2026-03-28*
*Authors: Greg Buscher (CTO), Claude Code (TC)*
