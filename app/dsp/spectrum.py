"""
Spectral Analysis DSP Module (Pure NumPy/SciPy Optimization)
Calculates FFT, spectral centroid, bandwidth, rolloff, contrast, flatness, and band energies
with zero Numba JIT overhead and minimal memory allocation.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class SpectralFeatures:
    spectral_centroid: float          # Average spectral center of mass (Hz)
    spectral_bandwidth: float         # Spectral spread (Hz)
    spectral_rolloff: float           # Frequency below which 85% energy lies (Hz)
    spectral_flatness: float          # 0 (pure tone) to 1 (pure white noise)
    spectral_contrast: List[float]    # Contrast across octave frequency bands
    dominant_frequencies: List[float] # Top dominant spectral peaks (Hz)
    low_energy_ratio: float           # Energy in 20 Hz - 250 Hz
    mid_energy_ratio: float           # Energy in 250 Hz - 4000 Hz
    high_energy_ratio: float          # Energy in 4000 Hz - 20000 Hz
    brightness: float                 # Normalized brightness metric [0.0, 1.0]


def analyze_spectrum(audio: np.ndarray, sr: int = 44100) -> SpectralFeatures:
    """Extract spectral features from mono audio array using pure vectorized NumPy."""
    if len(audio) == 0:
        return SpectralFeatures(
            spectral_centroid=1000.0,
            spectral_bandwidth=1000.0,
            spectral_rolloff=2000.0,
            spectral_flatness=0.5,
            spectral_contrast=[10.0] * 7,
            dominant_frequencies=[440.0],
            low_energy_ratio=0.33,
            mid_energy_ratio=0.33,
            high_energy_ratio=0.33,
            brightness=0.5
        )

    # Frame-based STFT via sliding window (capped to 4s for low memory and instant execution)
    analysis_audio = audio[:min(len(audio), sr * 4)]
    frame_len = min(2048, max(256, 2 ** int(np.floor(np.log2(len(analysis_audio))))))
    hop_len = max(256, frame_len // 2)

    n_frames = max(1, (len(analysis_audio) - frame_len) // hop_len + 1)
    frames = np.lib.stride_tricks.as_strided(
        analysis_audio,
        shape=(n_frames, frame_len),
        strides=(analysis_audio.strides[0] * hop_len, analysis_audio.strides[0])
    )
    win = np.hanning(frame_len).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * win, axis=-1))
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)

    # 1. Spectral Centroid
    spec_sum = np.sum(spec, axis=-1, keepdims=True) + 1e-12
    centroids = np.sum(spec * freqs, axis=-1, keepdims=True) / spec_sum
    mean_centroid = float(np.mean(centroids))

    # 2. Spectral Bandwidth
    bandwidths = np.sqrt(np.sum(spec * ((freqs - centroids) ** 2), axis=-1, keepdims=True) / spec_sum)
    mean_bandwidth = float(np.mean(bandwidths))

    # 3. Spectral Rolloff (85% energy)
    cum_energy = np.cumsum(spec ** 2, axis=-1)
    threshold = 0.85 * cum_energy[:, -1:]
    rolloff_bins = np.argmax(cum_energy >= threshold, axis=-1)
    mean_rolloff = float(np.mean(freqs[rolloff_bins]))

    # 4. Spectral Flatness (Wiener entropy: geometric mean / arithmetic mean)
    power = spec ** 2 + 1e-12
    geom_mean = np.exp(np.mean(np.log(power), axis=-1))
    arith_mean = np.mean(power, axis=-1)
    mean_flatness = float(np.clip(np.mean(geom_mean / (arith_mean + 1e-12)), 0.0, 1.0))

    # 5. Spectral Contrast across sub-bands
    # Octave bands: 0-200, 200-400, 400-800, 800-1600, 1600-3200, 3200-min(sr//2, 16000)
    band_edges = [0, 200, 400, 800, 1600, 3200, min(sr // 2, 16000)]
    contrast_vals = []
    mean_spec = np.mean(spec, axis=0)  # average across frames
    for b_idx in range(len(band_edges) - 1):
        low_f, high_f = band_edges[b_idx], band_edges[b_idx + 1]
        mask = (freqs >= low_f) & (freqs < high_f)
        if np.any(mask):
            b_power = mean_spec[mask]
            peak_val = np.percentile(b_power, 85) + 1e-6
            valley_val = np.percentile(b_power, 15) + 1e-6
            contrast_db = float(20.0 * np.log10(peak_val / valley_val))
            contrast_vals.append(round(contrast_db, 2))
        else:
            contrast_vals.append(10.0)

    # 6. Global FFT & Dominant Frequencies
    max_fft_len = min(len(audio), 4096)
    audio_slice = audio[:max_fft_len]
    fft_vals = np.abs(np.fft.rfft(audio_slice * np.hanning(len(audio_slice))))
    g_freqs = np.fft.rfftfreq(len(audio_slice), 1.0 / sr)

    # Find top 5 dominant peaks in 40Hz - 8000Hz
    valid_idx = np.where((g_freqs >= 40) & (g_freqs <= 8000))[0]
    if len(valid_idx) > 0:
        valid_fft = fft_vals[valid_idx]
        valid_freqs = g_freqs[valid_idx]

        top_indices = np.argsort(valid_fft)[::-1]
        dominant: List[float] = []
        for idx in top_indices:
            f = float(valid_freqs[idx])
            if all(abs(f - existing) > 0.05 * f for existing in dominant):
                dominant.append(round(f, 1))
            if len(dominant) >= 5:
                break
    else:
        dominant = [220.0, 440.0]

    # 7. Low / Mid / High Energy Ratios
    low_mask = (g_freqs >= 20) & (g_freqs < 250)
    mid_mask = (g_freqs >= 250) & (g_freqs < 4000)
    high_mask = (g_freqs >= 4000) & (g_freqs < 20000)

    low_pwr = np.sum(fft_vals[low_mask] ** 2) if np.any(low_mask) else 1e-6
    mid_pwr = np.sum(fft_vals[mid_mask] ** 2) if np.any(mid_mask) else 1e-6
    high_pwr = np.sum(fft_vals[high_mask] ** 2) if np.any(high_mask) else 1e-6
    total_pwr = low_pwr + mid_pwr + high_pwr + 1e-9

    low_ratio = float(low_pwr / total_pwr)
    mid_ratio = float(mid_pwr / total_pwr)
    high_ratio = float(high_pwr / total_pwr)

    # Normalized brightness index [0.0 = very dark/muddy, 1.0 = crisp/sparkling]
    brightness = float(np.clip((mean_centroid - 200.0) / 4800.0, 0.0, 1.0))

    return SpectralFeatures(
        spectral_centroid=round(mean_centroid, 2),
        spectral_bandwidth=round(mean_bandwidth, 2),
        spectral_rolloff=round(mean_rolloff, 2),
        spectral_flatness=round(mean_flatness, 4),
        spectral_contrast=contrast_vals,
        dominant_frequencies=dominant,
        low_energy_ratio=round(low_ratio, 3),
        mid_energy_ratio=round(mid_ratio, 3),
        high_energy_ratio=round(high_ratio, 3),
        brightness=round(brightness, 3)
    )
