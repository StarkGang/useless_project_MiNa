"""
Spectral Analysis DSP Module
Calculates FFT, spectral centroid, bandwidth, rolloff, contrast, flatness, and band energies.
"""

from dataclasses import dataclass
from typing import List, Tuple
import librosa
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
    """Extract spectral features from mono audio array using DSP."""
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

    # Precompute shared magnitude spectrogram (capped to 10s to prevent out-of-memory on 512MB RAM hosts)
    analysis_audio = audio[:min(len(audio), sr * 10)]
    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(len(analysis_audio))))))
    hop_length = 512
    S = np.abs(librosa.stft(analysis_audio, n_fft=n_fft, hop_length=hop_length))

    # 1. Spectral Centroid
    centroids = librosa.feature.spectral_centroid(S=S, sr=sr)
    mean_centroid = float(np.mean(centroids))

    # 2. Spectral Bandwidth
    bandwidths = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    mean_bandwidth = float(np.mean(bandwidths))

    # 3. Spectral Rolloff (85% energy)
    rolloffs = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    mean_rolloff = float(np.mean(rolloffs))

    # 4. Spectral Flatness (Wiener entropy)
    flatness = librosa.feature.spectral_flatness(S=S)
    mean_flatness = float(np.mean(flatness))

    # 5. Spectral Contrast across sub-bands
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)
    mean_contrast = [float(v) for v in np.mean(contrast, axis=1)]

    # 6. Global FFT & Dominant Frequencies
    # Limit FFT length for performance if audio is long
    max_fft_len = min(len(audio), sr * 10)
    audio_slice = audio[:max_fft_len]
    fft_vals = np.abs(np.fft.rfft(audio_slice * np.hanning(len(audio_slice))))
    freqs = np.fft.rfftfreq(len(audio_slice), 1.0 / sr)

    # Find top 5 dominant peaks in 40Hz - 8000Hz
    valid_idx = np.where((freqs >= 40) & (freqs <= 8000))[0]
    if len(valid_idx) > 0:
        valid_fft = fft_vals[valid_idx]
        valid_freqs = freqs[valid_idx]

        # Simple peak picking by sorting amplitude
        top_indices = np.argsort(valid_fft)[::-1]
        dominant: List[float] = []
        for idx in top_indices:
            f = float(valid_freqs[idx])
            # Ensure peaks are separated by at least half an octave
            if all(abs(f - existing) > 0.05 * f for existing in dominant):
                dominant.append(round(f, 1))
            if len(dominant) >= 5:
                break
    else:
        dominant = [220.0, 440.0]

    # 7. Low / Mid / High Energy Ratios
    # Low: 20-250 Hz, Mid: 250-4000 Hz, High: 4000-20000 Hz
    low_mask = (freqs >= 20) & (freqs < 250)
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = (freqs >= 4000) & (freqs < 20000)

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
        spectral_contrast=[round(c, 2) for c in mean_contrast],
        dominant_frequencies=dominant,
        low_energy_ratio=round(low_ratio, 3),
        mid_energy_ratio=round(mid_ratio, 3),
        high_energy_ratio=round(high_ratio, 3),
        brightness=round(brightness, 3)
    )
