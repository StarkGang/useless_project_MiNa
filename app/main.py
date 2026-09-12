"""
TheUnnecessaryFM Main FastAPI Application
Serves static frontend assets and REST API endpoints.
"""

import os
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes import router as api_router

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="TheUnnecessaryFM",
    description="Algorithmic Noise-to-Music System using pure DSP, audio analysis, and procedural synthesis.",
    version="1.0.0"
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
async def normalize_vercel_path(request: Request, call_next):
    """Normalize paths rewritten by Vercel serverless functions."""
    path = request.scope.get("path", "")
    for prefix in ("/api/index.py", "/index.py", "/api/index"):
        if path.startswith(prefix):
            new_path = path[len(prefix):] or "/"
            request.scope["path"] = new_path
            break
    return await call_next(request)


# Include API Router with and without /api prefix so Vercel rewrites work seamlessly
app.include_router(api_router, prefix="/api")
app.include_router(api_router, prefix="")


@app.get("/health")
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
