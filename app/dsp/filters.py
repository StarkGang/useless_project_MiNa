"""
DSP Digital Filters and Resonant Filter Bank Module
Implements biquad lowpass, highpass, bandpass, notch, and tuned resonant filter banks.
Turns arbitrary noise into tuned musical resonance.
"""

from typing import List, Tuple
import numpy as np
from scipy import signal


def design_biquad_lowpass(cutoff_hz: float, sr: int = 44100, q: float = 0.707) -> Tuple[np.ndarray, np.ndarray]:
    """Design a 2nd order Butterworth lowpass filter."""
    cutoff_hz = np.clip(cutoff_hz, 20.0, sr * 0.49)
    norm_cutoff = cutoff_hz / (sr * 0.5)
    b, a = signal.butter(2, norm_cutoff, btype='low')
    return b, a


def design_biquad_highpass(cutoff_hz: float, sr: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    """Design a 2nd order Butterworth highpass filter."""
    cutoff_hz = np.clip(cutoff_hz, 10.0, sr * 0.49)
    norm_cutoff = cutoff_hz / (sr * 0.5)
    b, a = signal.butter(2, norm_cutoff, btype='high')
    return b, a


def design_biquad_bandpass(center_hz: float, q: float = 10.0, sr: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a narrow 2nd order peaking/bandpass resonant filter.
    Q controls sharpness of the resonance (higher Q = narrower ringing tone).
    """
    center_hz = float(np.clip(center_hz, 20.0, sr * 0.48))
    bw = center_hz / max(1.0, q)
    f_low = max(10.0, center_hz - bw * 0.5)
    f_high = min(sr * 0.49, center_hz + bw * 0.5)
    b, a = signal.butter(2, [f_low / (sr * 0.5), f_high / (sr * 0.5)], btype='bandpass')
    return b, a


def apply_filter(audio: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply IIR filter using zero-phase filtering (filtfilt) or causal lfilter."""
    if len(audio) == 0:
        return audio
    # Use lfilter to preserve natural phase and avoid lookahead transient artifacting
    filtered = signal.lfilter(b, a, audio)
    return np.nan_to_num(filtered, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def resonant_filter_bank(
    audio: np.ndarray,
    frequencies: List[float],
    q: float = 24.0,
    sr: int = 44100,
    envelope_shaping: bool = True
) -> np.ndarray:
    """
    CRITICAL SECTION 8/17 REQUIREMENT:
    When there is NO pitch in the recording:
    Pass the source noise through a bank of tuned resonant bandpass filters (e.g. pentatonic scale notes).
    This turns non-musical noise (rain, static, fan, wind) into lush musical tones.
    """
    if len(audio) == 0 or not frequencies:
        return np.zeros_like(audio)

    resonated_sum = np.zeros_like(audio, dtype=np.float32)

    for freq in frequencies:
        if freq < 40.0 or freq > 8000.0:
            continue
        try:
            b, a = design_biquad_bandpass(center_hz=freq, q=q, sr=sr)
            # Filter noise
            filtered = apply_filter(audio, b, a)

            # Resonance gain compensation (resonance naturally attenuates broadband energy)
            boost = np.sqrt(q) * 2.5
            filtered = filtered * boost

            # Optional envelope shaping to prevent static continuous ringing
            if envelope_shaping:
                # Add gentle LFO tremolo (0.2 Hz to 0.8 Hz)
                t = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
                lfo = 0.7 + 0.3 * np.sin(2.0 * np.pi * (0.3 + 0.1 * (freq % 3)) * t)
                filtered = filtered * lfo

            resonated_sum += filtered
        except Exception:
            continue

    # Soft normalize resonated tone
    pk = np.max(np.abs(resonated_sum))
    if pk > 1e-4:
        resonated_sum = (resonated_sum / pk) * 0.85

    return resonated_sum.astype(np.float32)
