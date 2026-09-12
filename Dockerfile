FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    # Cap memory fragmentation on Linux container glibc
    MALLOC_ARENA_MAX=2 \
    # Cap OpenBLAS/OpenMP threads to 1 — prevents thread-thrashing on single-core free-tier CPUs
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1

# Install system audio dependencies and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition and install pure NumPy/SciPy audio stack
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn

# Pre-warm NumPy/SciPy vectorized math cache
RUN python -c "import numpy as np, scipy.signal; y = np.zeros(22050, dtype=np.float32); np.fft.rfft(y); b, a = scipy.signal.butter(2, 0.2); scipy.signal.lfilter(b, a, y); print('NumPy/SciPy DSP pre-warmed successfully')"

# Copy application source
COPY . .

# Create uploads and outputs folders
RUN mkdir -p uploads outputs

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log"]
