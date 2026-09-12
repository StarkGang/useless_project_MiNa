"""
Pitch Estimation and Tonal Salience DSP Module
Estimates fundamental frequencies via autocorrelation and harmonic salience.
Maps frequencies to musical note names, MIDI numbers, and builds a candidate pitch pool.
Detects pitch absence/confidence reliably.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import librosa
import numpy as np

# Note names for chromatic scale
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class PitchFeatures:
    has_reliable_pitch: bool         # True if clear tonal pitch detected
    pitch_confidence: float          # Confidence [0.0, 1.0]
    fundamental_hz: float            # Most salient fundamental frequency (Hz)
    nearest_note_name: str           # E.g. 'A4', 'C#3'
    nearest_midi: int                # MIDI note number (e.g. 69 for A4)
    candidate_notes: List[str]       # Dominant pitch classes detected (e.g. ['D', 'F', 'A'])
    candidate_midis: List[int]       # Dominant MIDI notes detected
    chroma_energy: List[float]       # 12-element chroma distribution (C through B)


def hz_to_midi(freq_hz):
    """Convert frequency in Hz to floating-point MIDI note number (A4 = 440 Hz = 69). Supports scalars and arrays."""
    if isinstance(freq_hz, (np.ndarray, list)):
        arr = np.asarray(freq_hz, dtype=np.float64)
        safe_arr = np.maximum(arr, 1e-4)
        res = 69.0 + 12.0 * np.log2(safe_arr / 440.0)
        res[arr <= 0] = 0.0
        return res
    if freq_hz <= 0:
        return 0.0
    return float(69.0 + 12.0 * np.log2(freq_hz / 440.0))


def midi_to_hz(midi_note):
    """Convert MIDI note number to frequency in Hz. Supports scalars and arrays."""
    if isinstance(midi_note, (np.ndarray, list)):
        arr = np.asarray(midi_note, dtype=np.float64)
        return 440.0 * (2.0 ** ((arr - 69.0) / 12.0))
    return float(440.0 * (2.0 ** ((midi_note - 69.0) / 12.0)))


def midi_to_note_name(midi_note: int) -> str:
    """Convert integer MIDI note number to note name string (e.g. 60 -> 'C4')."""
    octave = (midi_note // 12) - 1
    note_idx = midi_note % 12
    return f"{NOTE_NAMES[note_idx]}{octave}"


def analyze_pitch(audio: np.ndarray, sr: int = 44100) -> PitchFeatures:
    """
    Estimate fundamental pitch and tonal content using autocorrelation,
    harmonic salience, and 12-tone chromagram.
    """
    if len(audio) < 2048:
        return PitchFeatures(
            has_reliable_pitch=False,
            pitch_confidence=0.0,
            fundamental_hz=0.0,
            nearest_note_name="None",
            nearest_midi=60,
            candidate_notes=[],
            candidate_midis=[],
            chroma_energy=[1.0 / 12] * 12
        )

    # 1. Chromagram analysis (energy per pitch class across time)
    # Fast windowed chroma without heavy 2D HPSS median filtering
    analysis_chunk = audio[:min(len(audio), sr * 8)]
    chroma = librosa.feature.chroma_stft(y=analysis_chunk, sr=sr, n_fft=2048, hop_length=1024)
    chroma_mean = np.mean(chroma, axis=1)  # shape (12,)
    chroma_sum = np.sum(chroma_mean)
    if chroma_sum > 0:
        chroma_norm = chroma_mean / chroma_sum
    else:
        chroma_norm = np.ones(12) / 12.0

    # 2. Fundamental Frequency (F0) estimation via Autocorrelation & Harmonic Peaks
    # Analyze 2-second center window for fast execution on low-CPU containers
    center = len(audio) // 2
    chunk_len = min(len(audio), sr * 2)
    start = max(0, center - chunk_len // 2)
    chunk = audio[start:start + chunk_len]

    # Downsample chunk to 16kHz for fast and accurate pitch tracking (fmax is 1000 Hz)
    target_sr = 16000 if sr > 16000 else sr
    if target_sr != sr and len(chunk) > 0:
        chunk_ds = librosa.resample(chunk, orig_sr=sr, target_sr=target_sr)
    else:
        chunk_ds = chunk

    # Calculate frame-by-frame F0 using YIN/pYIN DSP algorithm (hop_length=1024 for 2x speedup)
    fmin = 55.0   # A1 (~55 Hz)
    fmax = 1000.0 # B5 (~987 Hz)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            chunk_ds,
            fmin=fmin,
            fmax=fmax,
            sr=target_sr,
            frame_length=1024,
            hop_length=1024
        )
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
        valid_probs = voiced_probs[voiced_flag & ~np.isnan(f0)]
    except Exception:
        valid_f0 = np.array([])
        valid_probs = np.array([])

    has_reliable_pitch = False
    fundamental_hz = 0.0
    pitch_confidence = 0.0

    if len(valid_f0) >= 4:
        # Weighted median F0
        fundamental_hz = float(np.median(valid_f0))
        mean_prob = float(np.mean(valid_probs)) if len(valid_probs) > 0 else 0.5
        # Measure pitch stability: standard deviation of semitones
        midis = hz_to_midi(valid_f0)
        semitone_std = float(np.std(midis))
        stability = float(np.clip(1.0 - (semitone_std / 4.0), 0.0, 1.0))
        pitch_confidence = float(np.clip(mean_prob * 0.7 + stability * 0.3, 0.0, 1.0))

        if pitch_confidence > 0.45 and 50.0 <= fundamental_hz <= 1200.0:
            has_reliable_pitch = True
    else:
        # No steady pitch detected
        fundamental_hz = 0.0
        pitch_confidence = 0.0
        has_reliable_pitch = False

    # Nearest note
    if has_reliable_pitch and fundamental_hz > 0:
        nearest_midi = int(round(hz_to_midi(fundamental_hz)))
        nearest_note_name = midi_to_note_name(nearest_midi)
    else:
        nearest_midi = 60
        nearest_note_name = "None"

    # Candidate notes from top 3 chroma peaks if significant
    top_chroma_indices = np.argsort(chroma_norm)[::-1]
    candidate_notes: List[str] = []
    candidate_midis: List[int] = []

    # If chroma has distinct peaks (max is noticeably above uniform 1/12 = 0.083)
    if chroma_norm[top_chroma_indices[0]] > 0.12:
        for idx in top_chroma_indices[:4]:
            if chroma_norm[idx] > 0.09:
                note_name = NOTE_NAMES[idx]
                candidate_notes.append(note_name)
                # Map to octave 4 for candidate MIDI
                candidate_midis.append(60 + idx)

    return PitchFeatures(
        has_reliable_pitch=has_reliable_pitch,
        pitch_confidence=round(pitch_confidence, 3),
        fundamental_hz=round(fundamental_hz, 1),
        nearest_note_name=nearest_note_name,
        nearest_midi=nearest_midi,
        candidate_notes=candidate_notes,
        candidate_midis=candidate_midis,
        chroma_energy=[round(float(v), 4) for v in chroma_norm]
    )
