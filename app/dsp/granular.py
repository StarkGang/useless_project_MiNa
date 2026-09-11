"""
Granular Synthesis and Time-Stretching DSP Module (High Performance)
Extracts grains, pitch shifts via fast linear interpolation, creates ambient pads,
and atmospheric soundscapes directly from source audio.
"""

from typing import List, Optional, Tuple
import numpy as np


def extract_grain(
    audio: np.ndarray,
    start_sample: int,
    grain_size: int,
    window_type: str = "hann"
) -> np.ndarray:
    """Extract a windowed grain from source audio."""
    end_sample = min(len(audio), start_sample + grain_size)
    actual_len = end_sample - start_sample
    if actual_len <= 0:
        return np.zeros(grain_size, dtype=np.float32)

    raw_grain = audio[start_sample:end_sample].copy()
    if actual_len < grain_size:
        raw_grain = np.pad(raw_grain, (0, grain_size - actual_len))

    win = np.hanning(grain_size).astype(np.float32)
    return (raw_grain * win).astype(np.float32)


def pitch_shift_grain(grain: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch shift a grain by resampling via fast linear interpolation."""
    if semitones == 0.0 or len(grain) == 0:
        return grain

    factor = 2.0 ** (-semitones / 12.0)
    new_len = max(16, int(round(len(grain) * factor)))

    # Fast linear interpolation
    x_old = np.linspace(0, 1, len(grain), endpoint=False)
    x_new = np.linspace(0, 1, new_len, endpoint=False)
    resampled = np.interp(x_new, x_old, grain).astype(np.float32)

    # Re-window to original grain length
    if len(resampled) > len(grain):
        resampled = resampled[:len(grain)]
    elif len(resampled) < len(grain):
        resampled = np.pad(resampled, (0, len(grain) - len(resampled)))

    win = np.hanning(len(resampled)).astype(np.float32)
    return (resampled * win).astype(np.float32)


def create_granular_pad(
    audio: np.ndarray,
    target_duration: float,
    semitone_shift: float = 0.0,
    grain_duration: float = 0.18,
    density: float = 18.0,
    sr: int = 44100,
    stereo_spread: bool = True
) -> np.ndarray:
    """
    Transforms source recording into a lush, sustained musical pad/drone
    via asynchronous granular cloud synthesis.
    Returns stereo array shape (2, N).
    """
    total_samples = int(target_duration * sr)
    grain_size = max(64, int(grain_duration * sr))

    left_out = np.zeros(total_samples, dtype=np.float32)
    right_out = np.zeros(total_samples, dtype=np.float32)

    if len(audio) < grain_size:
        audio = np.tile(audio, int(np.ceil(grain_size / max(1, len(audio)))))

    total_grains = int(target_duration * density)
    time_points = np.linspace(0, max(0, total_samples - grain_size), total_grains)

    rng = np.random.default_rng(42)

    for pos in time_points:
        int_pos = int(pos)
        src_start = rng.integers(0, max(1, len(audio) - grain_size))
        grain = extract_grain(audio, src_start, grain_size, window_type="hann")

        micro_detune = semitone_shift + float(rng.uniform(-0.12, 0.12))
        shifted = pitch_shift_grain(grain, micro_detune)

        if stereo_spread:
            pan = float(rng.uniform(0.15, 0.85))
            left_gain = float(np.cos(pan * np.pi * 0.5))
            right_gain = float(np.sin(pan * np.pi * 0.5))
        else:
            left_gain = 0.707
            right_gain = 0.707

        end_pos = min(total_samples, int_pos + grain_size)
        g_len = end_pos - int_pos
        left_out[int_pos:end_pos] += shifted[:g_len] * left_gain
        right_out[int_pos:end_pos] += shifted[:g_len] * right_gain

    # Normalize pad
    max_pk = max(float(np.max(np.abs(left_out))), float(np.max(np.abs(right_out))), 1e-5)
    left_out = (left_out / max_pk) * 0.75
    right_out = (right_out / max_pk) * 0.75

    # Gentle fade in and out (1.2s)
    fade_len = min(int(1.2 * sr), total_samples // 4)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    left_out[:fade_len] *= fade_in
    left_out[-fade_len:] *= fade_out
    right_out[:fade_len] *= fade_in
    right_out[-fade_len:] *= fade_out

    return np.stack([left_out, right_out], axis=0)


def create_source_texture(
    audio: np.ndarray,
    duration: float,
    sr: int = 44100,
    loop_points: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """Creates a continuous stereo texture layer with subtle breathing movement."""
    total_samples = int(duration * sr)
    if len(audio) == 0:
        return np.zeros((2, total_samples), dtype=np.float32)

    tile_len = len(audio)
    xfade_len = int(min(tile_len // 4, 0.1 * sr))
    reps = int(np.ceil(total_samples / max(1, tile_len - xfade_len))) + 1

    out = np.zeros(reps * tile_len, dtype=np.float32)
    ptr = 0
    for i in range(reps):
        if ptr + tile_len <= len(out):
            out[ptr:ptr + tile_len] += audio
        ptr += max(1, tile_len - xfade_len)

    out = out[:total_samples]
    delay_samples = int(0.012 * sr)
    left = out
    right = np.roll(out, delay_samples)
    right[:delay_samples] = 0.0

    return np.stack([left, right], axis=0).astype(np.float32)
