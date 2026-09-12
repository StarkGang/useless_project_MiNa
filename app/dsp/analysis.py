"""
Master Audio Analysis and Rule-Based Classification Module
Integrates Amplitude, Frequency, Rhythm, Harmonic, and Texture analysis.
Performs 100% rule-based classification into 9 sonic categories without any AI.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import librosa
import numpy as np

from .pitch import PitchFeatures, analyze_pitch
from .rhythm import RhythmFeatures, analyze_rhythm
from .spectrum import SpectralFeatures, analyze_spectrum


@dataclass
class AmplitudeFeatures:
    rms: float
    peak: float
    dynamic_range_db: float
    crest_factor: float
    envelope_curve: List[float]       # Downsampled 32-point RMS envelope curve for arrangement


@dataclass
class TextureFeatures:
    noisiness: float                  # Spectral flatness & high frequency ratio [0.0, 1.0]
    harmonicity: float                # Harmonic-to-noise ratio / harmonic energy ratio [0.0, 1.0]
    transient_density: float          # Transients per second
    temporal_entropy: float           # Variance of amplitude envelope


@dataclass
class CompleteAnalysis:
    classification: str               # One of MELODIC, HARMONIC, RHYTHMIC, PERCUSSIVE, TEXTURAL, VOCAL, AMBIENT, CHAOTIC, MIXED
    confidence: float                 # Classification confidence [0.0, 1.0]
    classification_reasons: List[str] # Explanation of rule-based logic
    amplitude: AmplitudeFeatures
    spectral: SpectralFeatures
    rhythm: RhythmFeatures
    pitch: PitchFeatures
    texture: TextureFeatures
    duration: float
    sr: int


def compute_amplitude_features(audio: np.ndarray, sr: int = 44100) -> AmplitudeFeatures:
    """Compute RMS, peak, dynamic range (dB), and downsampled envelope."""
    if len(audio) == 0:
        return AmplitudeFeatures(0.0, 0.0, 0.0, 1.0, [0.0] * 32)

    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Dynamic range in dB (peak to noise floor / RMS)
    if rms > 1e-7:
        crest_factor = float(peak / rms)
        dyn_range_db = float(20.0 * np.log10(max(1.0, crest_factor)))
    else:
        crest_factor = 1.0
        dyn_range_db = 0.0

    # 32-point normalized envelope curve across duration for compositional contour
    num_points = 32
    step = max(1, len(audio) // num_points)
    envelope = []
    for i in range(num_points):
        chunk = audio[i * step : min(len(audio), (i + 1) * step)]
        if len(chunk) > 0:
            env_val = float(np.sqrt(np.mean(chunk ** 2)))
        else:
            env_val = 0.0
        envelope.append(env_val)

    max_env = max(envelope) if envelope else 1e-6
    norm_envelope = [round(float(v / max(max_env, 1e-6)), 3) for v in envelope]

    return AmplitudeFeatures(
        rms=round(rms, 4),
        peak=round(peak, 4),
        dynamic_range_db=round(dyn_range_db, 2),
        crest_factor=round(crest_factor, 2),
        envelope_curve=norm_envelope
    )


def compute_texture_features(
    audio: np.ndarray,
    spec: SpectralFeatures,
    rhythm: RhythmFeatures,
    sr: int = 44100
) -> TextureFeatures:
    # Blazing fast FFT-based autocorrelation for harmonicity (avoids heavy O(N^2) CPU spike)
    try:
        sub_len = min(len(audio), sr * 1)
        sub = audio[:sub_len]
        if len(sub) > 0:
            n = len(sub)
            n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
            fx = np.fft.rfft(sub, n=n_fft)
            corr = np.fft.irfft(fx * np.conj(fx))[:n]
            min_lag = int(sr / 1200.0)
            max_lag = min(len(corr) - 1, int(sr / 60.0))
            if max_lag > min_lag and corr[0] > 1e-7:
                peak_val = float(np.max(corr[min_lag:max_lag]))
                harmonicity = float(np.clip(peak_val / corr[0], 0.0, 1.0))
            else:
                harmonicity = float(np.clip(1.0 - spec.spectral_flatness, 0.0, 1.0))
        else:
            harmonicity = 0.5
    except Exception:
        harmonicity = 0.5

    # Noisiness combines spectral flatness and high energy ratio
    noisiness = float(np.clip((spec.spectral_flatness * 1.5 + spec.high_energy_ratio * 0.5), 0.0, 1.0))

    # Temporal entropy (variance of 100ms frames)
    frame_size = int(0.1 * sr)
    num_frames = max(1, len(audio) // frame_size)
    frame_energies = [np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2) for i in range(num_frames)]
    temporal_entropy = float(np.std(frame_energies) / max(1e-6, np.mean(frame_energies)))

    return TextureFeatures(
        noisiness=round(noisiness, 3),
        harmonicity=round(harmonicity, 3),
        transient_density=round(rhythm.rhythmic_density, 2),
        temporal_entropy=round(min(5.0, temporal_entropy), 3)
    )


def classify_source(
    amp: AmplitudeFeatures,
    spec: SpectralFeatures,
    rhythm: RhythmFeatures,
    pitch: PitchFeatures,
    texture: TextureFeatures
) -> Tuple[str, float, List[str]]:
    """
    100% Rule-Based Classifier
    Categories:
      - MELODIC: clear pitched contour with low flatness and melodic movement
      - HARMONIC: stable chordal/tonal structure, strong harmonicity
      - RHYTHMIC: high periodicity, steady pulse, repetitive onsets
      - PERCUSSIVE: high transient density, high crest factor, short bursts
      - TEXTURAL: high spectral flatness, continuous drone or noise (rain, fan, static)
      - VOCAL: formant patterns in mid frequencies, pitch track in human speech range
      - AMBIENT: low dynamic range, smooth envelope, gentle spectral rolloff
      - CHAOTIC: high temporal entropy, non-periodic onsets, wide bandwidth
      - MIXED: balanced characteristics without a single dominant trait
    """
    reasons: List[str] = []

    # Rule 1: TEXTURAL (e.g. rain, fan, static, waterfall)
    if texture.noisiness > 0.45 or (spec.spectral_flatness > 0.15 and texture.harmonicity < 0.4):
        reasons.append(f"High spectral flatness ({spec.spectral_flatness}) and low harmonicity ({texture.harmonicity})")
        if spec.spectral_centroid < 800 and amp.crest_factor < 3.5:
            return "AMBIENT", 0.88, reasons + ["Low centroid and smooth crest factor indicate warm ambient drone"]
        return "TEXTURAL", 0.85, reasons

    # Rule 2: PERCUSSIVE (e.g. taps, knocks, clicks, typewriter, claps)
    if (amp.crest_factor > 4.5 or rhythm.rhythmic_density > 2.0) and texture.harmonicity < 0.35:
        reasons.append(f"High crest factor ({amp.crest_factor}) and low harmonicity ({texture.harmonicity})")
        if rhythm.has_reliable_rhythm:
            return "RHYTHMIC", 0.86, reasons + ["High transient density with periodic metronome alignment"]
        return "PERCUSSIVE", 0.84, reasons

    # Rule 3: RHYTHMIC (e.g. footsteps, clocks, rhythmic engine, beat)
    if rhythm.has_reliable_rhythm and rhythm.rhythmic_density >= 0.7:
        reasons.append(f"Strong periodicity ({rhythm.ioi_regularity}) and tempo confidence ({rhythm.tempo_confidence})")
        return "RHYTHMIC", 0.89, reasons

    # Rule 4: MELODIC / VOCAL / HARMONIC
    if pitch.has_reliable_pitch and pitch.pitch_confidence > 0.45:
        # Check if pitch is in human vocal fundamental range (85 Hz to 350 Hz) with mid-energy
        if 85.0 <= pitch.fundamental_hz <= 380.0 and spec.mid_energy_ratio > 0.55 and texture.temporal_entropy > 0.6:
            reasons.append(f"F0 ({pitch.fundamental_hz} Hz) in vocal range with dominant mid-frequency energy ({spec.mid_energy_ratio})")
            return "VOCAL", 0.82, reasons

        if texture.harmonicity > 0.6:
            reasons.append(f"High harmonic energy ({texture.harmonicity}) and stable pitch ({pitch.nearest_note_name})")
            return "HARMONIC", 0.85, reasons

        reasons.append(f"Pitch salience ({pitch.pitch_confidence}) with pitch class {pitch.nearest_note_name}")
        return "MELODIC", 0.80, reasons

    # Rule 5: AMBIENT (smooth, low transient, low dynamic variation)
    if rhythm.rhythmic_density < 0.5 and amp.crest_factor < 3.0 and texture.temporal_entropy < 0.4:
        reasons.append("Low transient density and smooth temporal envelope indicate ambient background")
        return "AMBIENT", 0.83, reasons

    # Rule 6: CHAOTIC (highly irregular, noisy spikes, erratic)
    if texture.temporal_entropy > 2.0 and not rhythm.has_reliable_rhythm and spec.spectral_bandwidth > 2500:
        reasons.append("High temporal entropy and erratic non-periodic energy distribution")
        return "CHAOTIC", 0.75, reasons

    # Fallback: MIXED
    reasons.append("Balanced blend of transient, textural, and tonal components")
    return "MIXED", 0.70, reasons


def analyze_audio(audio: np.ndarray, sr: int = 44100) -> CompleteAnalysis:
    """Run full DSP feature extraction suite and rule-based classifier."""
    amp = compute_amplitude_features(audio, sr=sr)
    spec = analyze_spectrum(audio, sr=sr)
    rhythm = analyze_rhythm(audio, sr=sr)
    pitch = analyze_pitch(audio, sr=sr)
    texture = compute_texture_features(audio, spec, rhythm, sr=sr)

    category, confidence, reasons = classify_source(amp, spec, rhythm, pitch, texture)

    return CompleteAnalysis(
        classification=category,
        confidence=round(confidence, 2),
        classification_reasons=reasons,
        amplitude=amp,
        spectral=spec,
        rhythm=rhythm,
        pitch=pitch,
        texture=texture,
        duration=round(len(audio) / sr, 2),
        sr=sr
    )
