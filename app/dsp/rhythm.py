"""
Rhythm and Transient Analysis DSP Module
Calculates onset strength, onset positions, inter-onset intervals, rhythmic regularity,
tempo estimation, and extracts transient slices for percussion.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class RhythmFeatures:
    estimated_tempo: float            # Estimated BPM (or fallback)
    has_reliable_rhythm: bool         # Whether strong periodic pulse is detected
    tempo_confidence: float           # Confidence [0.0, 1.0]
    onset_count: int                  # Total count of detected onsets
    rhythmic_density: float           # Onsets per second
    mean_ioi: float                   # Mean Inter-Onset Interval (seconds)
    ioi_regularity: float             # Regularity/periodicity metric [0.0 = chaotic, 1.0 = strict metronome]
    onset_times: List[float]          # List of onset timestamps (s)
    onset_samples: List[int]          # List of onset sample indices


def analyze_rhythm(audio: np.ndarray, sr: int = 44100) -> RhythmFeatures:
    """Analyze onset envelope, beat track, periodicity, and transient dynamics using pure vectorized DSP."""
    duration = len(audio) / sr
    if duration < 0.2:
        return RhythmFeatures(
            estimated_tempo=95.0,
            has_reliable_rhythm=False,
            tempo_confidence=0.0,
            onset_count=0,
            rhythmic_density=0.0,
            mean_ioi=0.0,
            ioi_regularity=0.0,
            onset_times=[],
            onset_samples=[]
        )

    # 1. Spectral flux onset envelope (pure NumPy, zero Numba JIT compiling)
    hop_length = 512
    n_fft = 1024
    analysis_audio = audio[:min(len(audio), sr * 60)]
    if len(analysis_audio) < n_fft:
        analysis_audio = np.pad(analysis_audio, (0, n_fft - len(analysis_audio)))

    n_frames = max(1, (len(analysis_audio) - n_fft) // hop_length + 1)
    frames = np.lib.stride_tricks.as_strided(
        analysis_audio,
        shape=(n_frames, n_fft),
        strides=(analysis_audio.strides[0] * hop_length, analysis_audio.strides[0])
    )
    win = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * win, axis=-1))

    # Half-wave rectified spectral novelty with high-frequency emphasis
    diff = np.diff(spec, axis=0)
    weights = np.linspace(0.8, 2.8, spec.shape[-1], dtype=np.float32)
    flux = np.sum(np.maximum(0.0, diff) * weights, axis=-1)

    max_flux = float(np.max(flux)) if len(flux) > 0 else 0.0
    if max_flux > 1e-6:
        onset_env = (flux / max_flux).astype(np.float32)
    else:
        onset_env = np.zeros(len(flux), dtype=np.float32)

    # 2. Onset peak detection with adaptive thresholding & 50ms refractory period
    onset_times: List[float] = []
    onset_samples: List[int] = []
    threshold = 0.12
    min_dist = max(1, int(0.05 * sr / hop_length))  # 50ms min distance

    peaks: List[int] = []
    for i in range(1, len(onset_env) - 1):
        if onset_env[i] > threshold and onset_env[i] > onset_env[i - 1] and onset_env[i] >= onset_env[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)

    onset_times = [round(float(p * hop_length / sr), 3) for p in peaks]
    onset_samples = [int(p * hop_length) for p in peaks]
    onset_count = len(onset_times)
    rhythmic_density = float(onset_count / max(0.5, duration))

    # 3. Inter-Onset Intervals (IOI) & Regularity
    if onset_count >= 3:
        iois = np.diff(onset_times)
        mean_ioi = float(np.mean(iois))
        std_ioi = float(np.std(iois))
        cov = std_ioi / max(mean_ioi, 1e-4)
        ioi_regularity = float(np.clip(1.0 - min(1.0, cov), 0.0, 1.0))
    else:
        mean_ioi = 0.0
        ioi_regularity = 0.0

    # 4. Tempo estimation via FFT-based autocorrelation of the onset envelope
    tempo_confidence = 0.0
    has_reliable_rhythm = False
    estimated_tempo = 95.0

    if len(onset_env) > 16:
        try:
            n_env = len(onset_env)
            n_fft_ac = 2 ** int(np.ceil(np.log2(2 * n_env - 1)))
            fx = np.fft.rfft(onset_env.astype(np.float64), n=n_fft_ac)
            ac = np.fft.irfft(fx * np.conj(fx))[:n_env]

            fps = sr / hop_length
            # Lag bounds corresponding to 60 BPM to 180 BPM
            min_lag = max(1, int(fps * 60.0 / 180.0))
            max_lag = min(len(ac) - 1, int(fps * 60.0 / 60.0))

            if max_lag > min_lag:
                ac_zone = ac[min_lag:max_lag]
                if len(ac_zone) > 0 and ac[0] > 1e-6:
                    norm_zone = ac_zone / ac[0]
                    best_rel = int(np.argmax(norm_zone))
                    best_lag = min_lag + best_rel
                    peak_ac = float(norm_zone[best_rel])

                    raw_bpm = (fps * 60.0) / best_lag
                    bpm = raw_bpm
                    while bpm > 140.0:
                        bpm /= 2.0
                    while bpm < 65.0:
                        bpm *= 2.0

                    # Periodic rhythm requires consistent inter-onset intervals (ioi_regularity)
                    tempo_confidence = float(np.clip((peak_ac * 0.5) + (ioi_regularity * 0.5), 0.0, 1.0))
                    if tempo_confidence > 0.38 and rhythmic_density > 0.6 and ioi_regularity >= 0.18:
                        has_reliable_rhythm = True
                        estimated_tempo = round(bpm, 1)
                    else:
                        estimated_tempo = round(bpm, 1) if (70 <= bpm <= 130 and ioi_regularity >= 0.15) else 95.0
        except Exception:
            estimated_tempo = 95.0

    return RhythmFeatures(
        estimated_tempo=float(estimated_tempo),
        has_reliable_rhythm=has_reliable_rhythm,
        tempo_confidence=round(tempo_confidence, 3),
        onset_count=onset_count,
        rhythmic_density=round(rhythmic_density, 2),
        mean_ioi=round(mean_ioi, 4),
        ioi_regularity=round(ioi_regularity, 3),
        onset_times=onset_times,
        onset_samples=onset_samples
    )


def extract_transient_slices(
    audio: np.ndarray,
    onset_samples: List[int],
    sr: int = 44100,
    max_slices: int = 8,
    slice_duration: float = 0.35
) -> List[np.ndarray]:
    """
    Slice the source audio around detected onsets to produce percussion samples.
    Normalizes each slice and applies a fast fade-out envelope to eliminate clicks.
    """
    slices: List[np.ndarray] = []
    slice_len = int(slice_duration * sr)
    fade_out_len = int(0.02 * sr)
    fade_window = np.linspace(1.0, 0.0, fade_out_len)

    # Sort onsets by local energy to pick the crispiest transients
    scored_onsets = []
    for s in onset_samples:
        if s + 256 < len(audio):
            local_energy = float(np.sum(audio[s:min(len(audio), s + 1024)] ** 2))
            scored_onsets.append((local_energy, s))

    scored_onsets.sort(key=lambda x: x[0], reverse=True)

    for _, s in scored_onsets[:max_slices]:
        end = min(len(audio), s + slice_len)
        slice_audio = audio[s:end].copy()

        # Apply fade out at tail
        if len(slice_audio) > fade_out_len:
            slice_audio[-fade_out_len:] *= fade_window
        elif len(slice_audio) > 0:
            slice_audio *= np.linspace(1.0, 0.0, len(slice_audio))

        # Peak normalize slice
        pk = np.max(np.abs(slice_audio))
        if pk > 1e-4:
            slice_audio = (slice_audio / pk) * 0.95

        slices.append(slice_audio.astype(np.float32))

    return slices
