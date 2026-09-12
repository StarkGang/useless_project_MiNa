"""
TheUnnecessaryFM Main FastAPI Application
Serves static frontend assets and REST API endpoints.
"""

import os

# Thread limits and memory flags to ensure optimal execution on single-core / low-resource instances
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('BLIS_NUM_THREADS', '1')
os.environ.setdefault('MALLOC_ARENA_MAX', '2')

from contextlib import asynccontextmanager
import threading
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes import router as api_router

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


def _warmup_dsp():
    """Warm up NumPy FFT and SciPy filter caches during startup."""
    try:
        import numpy as np
        import scipy.signal
        dummy = np.zeros(1024, dtype=np.float32)
        np.fft.rfft(dummy)
        b, a = scipy.signal.butter(2, 0.2, btype='low')
        scipy.signal.lfilter(b, a, dummy)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Non-blocking warm-up in background thread at startup
    t = threading.Thread(target=_warmup_dsp, daemon=True)
    t.start()
    yield


app = FastAPI(
    title="TheUnnecessaryFM",
    description="Algorithmic Noise-to-Music System using pure DSP, audio analysis, and procedural synthesis.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_file_size_limit_header(request: Request, call_next):
    """Early rejection if Content-Length header exceeds the configured file size limit."""
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                from .api.routes import get_file_size_limit_bytes
                max_bytes = get_file_size_limit_bytes()
                # Content-Length in multipart/form-data includes form boundaries, field names, and headers (~64KB margin)
                if int(content_length) > (max_bytes + 65536):
                    import json
                    limit_mb = max_bytes / (1024 * 1024)
                    return Response(
                        content=json.dumps({
                            "detail": f"Request payload size exceeds the allowed limit of {limit_mb:.1f}MB."
                        }),
                        status_code=413,
                        media_type="application/json"
                    )
            except Exception:
                pass
    return await call_next(request)


@app.middleware("http")
async def normalize_vercel_path(request: Request, call_next):
    """Normalize paths rewritten by Vercel serverless functions."""
    # 1. Query parameter __path__ from Vercel rewrite
    target = request.query_params.get("__path__")
    if target:
        clean = target if target.startswith("/") else f"/{target}"
        request.scope["path"] = f"/api{clean}" if not clean.startswith("/api") else clean
        return await call_next(request)

    # 2. Check x-now-route-matches (Vercel sets this for regex rewrites e.g. "1=generate")
    route_matches = request.headers.get("x-now-route-matches")
    if route_matches:
        import urllib.parse
        for part in route_matches.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                if k in ("1", "path", "match"):
                    val = urllib.parse.unquote(v)
                    clean = val if val.startswith("/") else f"/{val}"
                    request.scope["path"] = f"/api{clean}" if not clean.startswith("/api") else clean
                    return await call_next(request)

    # 3. Check x-matched-path if it represents the user-requested URL (not just /api)
    matched = request.headers.get("x-matched-path")
    if matched and matched not in ("/api", "/api/", "/api/index", "/api/index.py") and not matched.endswith(".py"):
        request.scope["path"] = matched
        return await call_next(request)

    # 4. Standard prefix strip if path begins with /api/index.py or /index.py
    path = request.scope.get("path", "")
    for prefix in ("/api/index.py", "/index.py", "/api/index"):
        if path.startswith(prefix):
            remainder = path[len(prefix):]
            if remainder:
                request.scope["path"] = remainder if remainder.startswith("/") else f"/{remainder}"
                return await call_next(request)
            break

    # 5. Fallback: If hitting /api/index.py directly via POST multipart, route to /api/generate
    if path in ("/api/index.py", "/index.py", "/api/index", "/api", ""):
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if "multipart/form-data" in content_type:
                request.scope["path"] = "/api/generate"
                return await call_next(request)
        elif request.method == "GET" and path in ("/api/index.py", "/index.py", "/api/index"):
            request.scope["path"] = "/health"
            return await call_next(request)

    return await call_next(request)


# Include API Router with and without /api prefix so Vercel rewrites work seamlessly
app.include_router(api_router, prefix="/api")
app.include_router(api_router, prefix="")


@app.get("/health")
@app.get("/api/health")
@app.get("/api")
def health_check():
    """Fast health check endpoint for monitoring and keep-alive pingers."""
    return {"status": "ok", "service": "TheUnnecessaryFM"}


@app.get("/env.js")
def get_env_js():
    """Serves runtime environment variables (BACKEND_URL) as JavaScript to frontend."""
    backend_url = (
        os.environ.get("BACKEND_URL")
        or os.environ.get("API_BASE_URL")
        or ""
    )
    content = f'window.ENV = {{ BACKEND_URL: "{backend_url}" }};'
    return Response(content=content, media_type="application/javascript")


# Mount Frontend static files only when running locally / non-serverless
is_serverless = bool(
    os.environ.get("VERCEL")
    or os.environ.get("VERCEL_ENV")
    or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
)

if FRONTEND_DIR.exists() and not is_serverless:
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
