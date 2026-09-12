"""
Audio Input Processing for TheUnnecessaryFM
Decodes any audio format via FFmpeg, resamples to 44.1 kHz, cleans DC offset,
preserves stereo and provides mono for analysis, detects silence and clipping.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import imageio_ffmpeg
import numpy as np
import soundfile as sf

TARGET_SR = 44100


@dataclass
class PreprocessedAudio:
    mono: np.ndarray          # 1D float32 array, normalized [-1.0, 1.0]
    stereo: np.ndarray        # 2D float32 array, shape (2, N)
    sr: int                   # Sample rate (always 44100)
    duration: float           # Duration in seconds
    is_silent: bool           # Whether signal is essentially silence
    is_clipped: bool          # Whether original signal exhibited clipping
    clipping_ratio: float     # Percentage of samples clipped
    dc_offset: float          # Measured DC offset before removal
    peak_before_norm: float   # Peak amplitude prior to normalization
    original_channels: int    # Channels in source file


def get_ffmpeg_binary() -> str:
    """Locate the FFmpeg binary using imageio_ffmpeg or system PATH."""
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    raise RuntimeError("FFmpeg executable not found. Please install ffmpeg or imageio-ffmpeg.")


def decode_to_wav(input_path: str, output_wav_path: str, target_sr: int = TARGET_SR) -> None:
    """Decode any input container (WAV, MP3, WebM, OGG, etc.) to 44.1kHz WAV via FFmpeg."""
    ffmpeg_exe = get_ffmpeg_binary()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ar", str(target_sr),
        "-acodec", "pcm_f32le",
        str(output_wav_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err_msg = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg decoding failed: {err_msg}")


def preprocess_audio(path: str, target_sr: int = TARGET_SR) -> PreprocessedAudio:
    """
    Main preprocessing pipeline:
    1. Decode using FFmpeg.
    2. Convert to WAV.
    3. Resample to 44.1 kHz.
    4. Convert to mono for analysis.
    5. Preserve stereo source when available.
    6. Normalize safely.
    7. Remove DC offset.
    8. Detect silence.
    9. Detect clipping.
    10. Calculate source duration.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    # Fast path: Try direct reading via soundfile if already a clean WAV at target_sr
    data = None
    try:
        raw_data, file_sr = sf.read(str(path_obj), dtype="float32", always_2d=True)
        if file_sr == target_sr and raw_data.shape[0] > 0:
            data = raw_data
            sr = file_sr
    except Exception:
        data = None

    if data is None:
        # Fallback to FFmpeg transcoding for non-WAV formats (MP3, OGG, WebM) or different sample rates
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_wav_path = tmp.name

        try:
            decode_to_wav(str(path_obj), temp_wav_path, target_sr=target_sr)
            data, sr = sf.read(temp_wav_path, dtype="float32", always_2d=True)
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except OSError:
                    pass

    num_samples, orig_channels = data.shape

    # Check for clipping before any alterations
    abs_data = np.abs(data)
    clipping_samples = np.sum(abs_data >= 0.999)
    clipping_ratio = float(clipping_samples / max(1, data.size))
    is_clipped = clipping_ratio > 0.0005

    # Peak before normalization
    peak_before_norm = float(np.max(abs_data)) if data.size > 0 else 0.0

    # Remove DC offset per channel
    dc_offset = float(np.mean(data))
    data = data - np.mean(data, axis=0, keepdims=True)

    # Convert to mono for analysis
    if orig_channels > 1:
        mono = np.mean(data, axis=1)
        # Stereo preservation
        if orig_channels == 2:
            stereo = data.T  # shape (2, N)
        else:
            stereo = np.stack([mono, mono], axis=0)
    else:
        mono = data[:, 0]
        stereo = np.stack([mono, mono], axis=0)

    # Detect silence
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size > 0 else 0.0
    is_silent = rms < 1e-4 or peak_before_norm < 1e-4

    # Safe normalization: normalize with 1.0 dB headroom (target peak ~0.891)
    if not is_silent and peak_before_norm > 0:
        target_peak = 0.891
        gain = target_peak / max(peak_before_norm, 1e-6)
        # Avoid excessive amplification of pure background hiss
        gain = min(gain, 15.0)
        mono = np.clip(mono * gain, -1.0, 1.0)
        stereo = np.clip(stereo * gain, -1.0, 1.0)
    elif is_silent:
        # If input is pure silence, generate very faint warm acoustic noise floor to allow algorithmic processing
        noise = np.random.uniform(-0.001, 0.001, size=len(mono)).astype(np.float32)
        mono = mono + noise
        stereo = stereo + np.stack([noise, noise], axis=0)

    duration = float(len(mono) / target_sr)

    return PreprocessedAudio(
        mono=mono.astype(np.float32),
        stereo=stereo.astype(np.float32),
        sr=target_sr,
        duration=duration,
        is_silent=is_silent,
        is_clipped=is_clipped,
        clipping_ratio=clipping_ratio,
        dc_offset=dc_offset,
        peak_before_norm=peak_before_norm,
        original_channels=orig_channels
    )
