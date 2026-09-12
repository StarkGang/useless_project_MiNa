"""
Rhythm and Transient Analysis DSP Module
Calculates onset strength, onset positions, inter-onset intervals, rhythmic regularity,
tempo estimation, and extracts transient slices for percussion.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import librosa
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
    """Analyze onset envelope, beat track, periodicity, and transient dynamics."""
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

    # 1. Onset strength envelope
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)

    # 2. Onset peak detection
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        delta=0.07
    )
    onset_times = [float(t) for t in librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)]
    onset_samples = [int(s) for s in librosa.frames_to_samples(onset_frames, hop_length=hop_length)]

    onset_count = len(onset_times)
    rhythmic_density = float(onset_count / max(0.5, duration))

    # 3. Inter-Onset Intervals (IOI) & Regularity
    if onset_count >= 3:
        iois = np.diff(onset_times)
        mean_ioi = float(np.mean(iois))
        std_ioi = float(np.std(iois))
        # Coefficient of variation (lower = more regular)
        cov = std_ioi / max(mean_ioi, 1e-4)
        # Invert cov to regularity score [0.0, 1.0]
        ioi_regularity = float(np.clip(1.0 - min(1.0, cov), 0.0, 1.0))
    else:
        mean_ioi = 0.0
        ioi_regularity = 0.0

    # 4. Tempo estimation via beat tracking and autocorrelation
    tempo_confidence = 0.0
    has_reliable_rhythm = False
    estimated_tempo = 95.0

    try:
        tempo_fn = getattr(librosa.feature, "tempo", getattr(librosa.beat, "tempo", None))
        if tempo_fn is not None:
            tempo_result = tempo_fn(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop_length,
                aggregate=None
            )
        else:
            tempo_result = None
        if tempo_result is not None and len(tempo_result) > 0:
            bpm = float(np.median(tempo_result))
            # Keep BPM in musical range [60, 140], halving or doubling if necessary
            while bpm > 140.0:
                bpm /= 2.0
            while bpm < 65.0:
                bpm *= 2.0

            # Confidence based on onset strength variance and autocorrelation peak
            ac = librosa.autocorrelate(onset_env, max_size=int(sr / hop_length * 4))
            if len(ac) > 1:
                ac_norm = ac[1:] / (ac[0] + 1e-8)
                peak_ac = float(np.max(ac_norm)) if len(ac_norm) > 0 else 0.0
            else:
                peak_ac = 0.0

            tempo_confidence = float(np.clip((peak_ac * 0.6) + (ioi_regularity * 0.4), 0.0, 1.0))
            if tempo_confidence > 0.45 and rhythmic_density > 0.8:
                has_reliable_rhythm = True
                estimated_tempo = round(bpm, 1)
            else:
                # Default to a nice musical tempo when rhythm is ambiguous
                estimated_tempo = round(bpm, 1) if 70 <= bpm <= 130 else 95.0
    except Exception:
        estimated_tempo = 95.0
        tempo_confidence = 0.1
        has_reliable_rhythm = False

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
