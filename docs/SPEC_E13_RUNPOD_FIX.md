# SPEC: E13 RunPod Final Fix
**Document:** TC_BUILD_E13_RUNPOD_FIX_2026-03-29
**Author:** Architect (Claude Opus) for CTO Greg Buscher
**Repo:** ctt-vision-service
**Branch:** main (currently at `ccd9b9a`)
**Priority:** P1 — One commit to ship E13
**Estimated TC Time:** 10-15 minutes

---

## OBJECTIVE

Ship E13: Get the CTT Vision Service running on RunPod serverless. The architecture is correct (validated in RUNPOD_DEPLOYMENT_LESSONS.md). Three gaps remain that are likely causing the "context deadline exceeded" failure on container create.

---

## CONTEXT (What TC Already Solved)

TC fixed 8 issues across 11 commits on 3/28. The current architecture is sound:
- Docker image: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` + PyTorch 2.4.1 upgrade
- Model: Qwen2.5-VL-7B-Instruct on Network Volume (`ctt-vision-models`, US-NC-1, 20GB)
- Handler: crash-proof wrapper, health check endpoint, direct model loader
- Image size: ~6GB (model NOT baked in)

The last test hit "context deadline exceeded" on container create. Build `ccd9b9a` completed and pushed to registry but was never successfully tested on a fresh worker.

---

## THREE FIXES (This Commit)

### Fix 1: Add RUNPOD_INIT_TIMEOUT Environment Variable

**Problem:** RunPod's default init timeout is ~7 minutes. The lessons doc states cold starts take 5-10 minutes (image pull + model load). On a slow pull day, 7 minutes is exceeded and RunPod kills the container before the handler registers.

**Action:** This is a RunPod Dashboard change, NOT a code change. But document it in the repo.

**RunPod Dashboard → Serverless → ctt-vision-service → Edit Endpoint → Environment Variables:**
```
RUNPOD_INIT_TIMEOUT=600
```

This gives the container 10 minutes to initialize. If model loads in 18.4s and image pull takes 5 min worst case, 10 minutes is plenty of headroom.

**Also add to Dockerfile as a default (belt and suspenders):**
```dockerfile
ENV RUNPOD_INIT_TIMEOUT=600
```

### Fix 2: Pin transformers to Exact Version

**Problem:** Commit `ccd9b9a` bumped `requirements-runpod.txt` to `transformers>=5.0`. This is an open-ended floor that lets pip resolve to whatever the latest release is at build time. If transformers 5.x introduced breaking changes to Qwen2.5-VL model loading, the container crashes during init.

The lessons doc (which documents the working local test) specifies `transformers>=4.45.0` — a much safer floor.

**Action:** In `requirements-runpod.txt`, replace:
```
# BEFORE
transformers>=5.0

# AFTER
transformers>=4.45.0,<5.0
```

This keeps the version range that was validated during the successful RTX 4090 pod test. If `transformers>=5.0` is actually required for a specific reason (document it), then pin exact: `transformers==5.0.0`.

### Fix 3: Handle decord Dependency

**Problem:** The Arch Quick Ref v1.9 lists `decord` as one of 5 root causes fixed on 3/28, but the lessons doc doesn't mention it and it's not in `requirements-runpod.txt`. Qwen2.5-VL uses `decord` for video frame extraction via `qwen-vl-utils`. If `qwen-vl-utils` tries to import `decord` at module load time and it's not installed, the handler crashes before registering.

**Action:** Add `decord` to `requirements-runpod.txt`:
```
decord>=0.6.0
```

**OR** if decord causes its own install issues (it needs ffmpeg libs), add a defensive import guard in `runpod_handler.py` or `runpod_model_loader.py` wherever `qwen-vl-utils` is imported:

```python
try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    if "decord" in str(e):
        # decord not needed for image-only inference
        import importlib
        import sys
        # Stub decord so qwen_vl_utils doesn't crash
        sys.modules['decord'] = type(sys)('decord')
        sys.modules['decord'].VideoReader = None
        sys.modules['decord'].cpu = lambda: None
        from qwen_vl_utils import process_vision_info
    else:
        raise
```

**Preferred approach:** Try adding `decord>=0.6.0` to requirements first. Only use the stub if decord fails to install in the container (missing system libs).

If decord requires system dependencies, add to Dockerfile before pip install:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
```

---

## UPDATED DOCKERFILE (Complete — Replace Existing)

```dockerfile
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps (git for pip installs, ffmpeg for decord/video processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade PyTorch to 2.4+ (transformers requires it for Qwen2.5-VL)
RUN pip install --no-cache-dir torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir -r requirements-runpod.txt

# Copy handler code
COPY runpod_handler.py .
COPY runpod_model_loader.py .
COPY schemas/ ./schemas/

# Environment
ENV HF_HOME=/runpod-volume/hf_cache
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_INIT_TIMEOUT=600

CMD ["python", "-u", "runpod_handler.py"]
```

---

## UPDATED REQUIREMENTS (requirements-runpod.txt — Replace Existing)

```
runpod>=1.7.0
transformers>=4.45.0,<5.0
accelerate>=0.34.0
Pillow>=10.0.0
qwen-vl-utils>=0.0.8
decord>=0.6.0
```

**DO NOT include `torch` or `torchvision`** — handled in Dockerfile.

---

## DEPLOYMENT STEPS (TC Execute in Order)

### Step 1: Apply Code Changes
1. Update `Dockerfile` with the version above
2. Update `requirements-runpod.txt` with the version above
3. If `runpod_handler.py` imports `qwen_vl_utils` at top level, add the defensive import guard from Fix 3

### Step 2: Build and Push
```bash
docker build -t gbuscherinak/ctt-vision-service:latest .
docker push gbuscherinak/ctt-vision-service:latest
```

**Note:** If the Docker Hub org/username is different, adjust accordingly. Use whatever registry the endpoint is currently pointed at.

### Step 3: RunPod Dashboard Configuration
1. Go to RunPod Dashboard → Serverless → ctt-vision-service
2. Click Edit Endpoint
3. Under Environment Variables, add:
   - `RUNPOD_INIT_TIMEOUT` = `600`
4. Verify existing env vars:
   - `HF_HOME` = `/runpod-volume/hf_cache`
   - `PYTHONUNBUFFERED` = `1`
5. Verify GPU selection includes GPUs available in **US-NC-1** (where Network Volume lives)
6. Verify Network Volume `ctt-vision-models` is attached
7. Verify Max Workers = 1, Min Workers = 0
8. Verify Idle Timeout >= 60 seconds
9. Save changes — this triggers a new deployment with the latest image

### Step 4: Wait for Build
RunPod will pull the new image. First pull to the data center may take 10-20 minutes. Be patient. Check the endpoint status page for build progress.

### Step 5: Test Health Check
```bash
curl -X POST https://api.runpod.ai/v2/2vqdjupzh317p4/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"input":{"action":"health"}}'
```

**Expected response:** `{"status":"COMPLETED","output":{"status":"ok",...}}`

If health check returns successfully → proceed to Step 6.
If timeout → check container logs in RunPod dashboard for the actual error.

### Step 6: Test Real Vision Request
```bash
curl -X POST https://api.runpod.ai/v2/2vqdjupzh317p4/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"input":{"action":"classify","image_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg","schema":"theo_classify"}}'
```

Adjust the payload to match whatever your handler expects. The point is to confirm the model loads and runs inference.

**If this returns a valid classification → E13 SHIPPED.**

### Step 7: Post-Ship Cleanup
1. Terminate any running test pods on RunPod (check billing)
2. Set Max Workers back to 0 if not actively using (or leave at 1 for testing)
3. Verify endpoint idles down after 60s
4. Update RUNPOD_DEPLOYMENT_LESSONS.md with the init timeout fix
5. Commit all changes

---

## COMMIT MESSAGE

```
E13: RunPod final fix — init timeout + pinned transformers + decord

- Added RUNPOD_INIT_TIMEOUT=600 to Dockerfile ENV (also set in dashboard)
- Pinned transformers>=4.45.0,<5.0 (was >=5.0, untested bleeding edge)
- Added decord>=0.6.0 for qwen-vl-utils video dependency
- Added ffmpeg + libsm6 + libxext6 system deps for decord
- Updated Dockerfile with complete working configuration

Fixes "context deadline exceeded" on container create.
Closes E13.
```

---

## IF IT STILL FAILS

### Scenario A: Health check times out even with RUNPOD_INIT_TIMEOUT=600
- Increase to `RUNPOD_INIT_TIMEOUT=900` (15 minutes)
- Check if image is still pulling (RunPod dashboard → endpoint → workers tab)
- If image pull takes >10 min, the image may be too large. Check `docker images` for actual size

### Scenario B: Health check returns but model load fails
- Check if Network Volume is mounted: handler should log `/runpod-volume/` contents
- Check if model files exist at `/runpod-volume/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/`
- If missing, re-cache: deploy a temp Pod with the volume, re-download the model

### Scenario C: Container starts but crashes on import
- Check container logs for Python import errors
- Most likely: `decord` or `qwen-vl-utils` import failure
- Apply the defensive import stub from Fix 3
- Rebuild and redeploy

### Scenario D: GPU not available in US-NC-1
- RunPod may not have 24GB PRO GPUs available in that data center at that moment
- Try: remove 48GB PRO from GPU selection (less demand = faster allocation)
- Or: wait and retry — GPU availability fluctuates

---

## FILES CHANGED

| File | Action |
|------|--------|
| Dockerfile | REPLACE (complete new version above) |
| requirements-runpod.txt | REPLACE (complete new version above) |
| runpod_handler.py | MODIFY (add decord import guard if needed) |
| RUNPOD_DEPLOYMENT_LESSONS.md | UPDATE (add init timeout lesson) |

---

## REFERENCE

- RunPod Endpoint ID: `2vqdjupzh317p4`
- Network Volume: `ctt-vision-models` (US-NC-1, 20GB)
- Docker Registry: (whatever registry current endpoint points to)
- Previous working local test: RTX 4090, model loads in 18.4s
- RunPod Balance: $199.98

---

*Spec prepared by Architect (Claude Opus) | 2026-03-29*
*For TC execution. CTO approved.*
