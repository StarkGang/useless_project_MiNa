"""
Master Audio Analysis and Rule-Based Classification Module
Integrates Amplitude, Frequency, Rhythm, Harmonic, and Texture analysis.
Performs 100% rule-based classification into 9 sonic categories without any AI.
"""

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
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
class SmartSourceProfile:
    primary_category: str             # Dominant category (VOCAL, HUMMING, BEATBOX, TRAFFIC, PERCUSSIVE, etc.)
    display_title: str                # E.g. "Human Voice & Speech", "Vocal Beatbox Kit", "Composite Audio (Voice + Beatbox + Traffic)"
    has_speech_or_vocal: bool         # Spoken word, lyrics, formants detected
    has_humming: bool                 # Sustained vocal hum or whistling detected
    has_beatbox: bool                 # Vocal percussion, mouth beats, plosive transients detected
    has_traffic_or_engine: bool       # Low engine rumble, tire wash, traffic or horns detected
    has_foley_percussive: bool        # Crisp taps, clicks, claps detected
    has_ambient_bed: bool             # Atmospheric texture, rain, air bed detected
    autotune_enabled: bool            # True if voice, speech, humming, or tonal content should be auto-tuned
    autotune_strength: float          # 0.0 to 1.0 (1.0 = hard snap, 0.7 = natural soul)
    detected_elements: List[str]      # List of all detected sonic elements
    smart_settings_summary: str       # Clear, user-friendly description of the custom DSP settings applied
    recommended_styles: List[str]     # Best suited musical genres for this sound


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
    smart_profile: Optional[SmartSourceProfile] = None


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

    # Rule 1.5: TEXTURAL / URBAN ENVIRONMENTAL (e.g. street traffic, road noise, urban wash)
    if not pitch.has_reliable_pitch and spec.spectral_bandwidth > 2200.0 and (rhythm.rhythmic_density > 1.8 or len(rhythm.onset_samples) > 15):
        reasons.append(f"Broad continuous bandwidth ({spec.spectral_bandwidth:.0f} Hz) and dense non-periodic field transients ({rhythm.rhythmic_density:.1f}/s)")
        return "TEXTURAL", 0.88, reasons

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


def detect_smart_source_profile(
    audio: np.ndarray,
    amp: AmplitudeFeatures,
    spec: SpectralFeatures,
    rhythm: RhythmFeatures,
    pitch: PitchFeatures,
    texture: TextureFeatures,
    category: str,
    sr: int = 44100
) -> SmartSourceProfile:
    """
    Intelligently analyzes the source audio for multiple overlapping acoustic characteristics:
    Human Speech, Vocal Humming, Vocal Beatbox, Urban Traffic/Engine, Percussive Foley, and Ambient Beds.
    Handles composite/hybrid recordings containing some or all of these elements simultaneously.
    """
    detected_elements: List[str] = []
    settings_parts: List[str] = []
    styles: List[str] = []

    # 1. Detect Traffic / Urban / Engine Noise
    has_traffic_or_engine = False
    # Engine / heavy vehicle low-frequency rumble profile (sub rumble, engine idle, low hum)
    is_engine_rumble = (
        (spec.low_energy_ratio > 0.35)
        and (not pitch.has_reliable_pitch or pitch.fundamental_hz < 75.0 or pitch.pitch_confidence < 0.45)
        and amp.crest_factor < 13.0
    )
    # Phone mic high-pass rolls off sub-150Hz rumble, but traffic/road audio has wide bandwidth, high onset density, unpitched
    is_urban_traffic = (
        (not pitch.has_reliable_pitch or pitch.pitch_confidence < 0.35)
        and spec.spectral_bandwidth > 2200.0
        and (len(rhythm.onset_samples) >= 15 or rhythm.rhythmic_density >= 1.5)
        and (spec.mid_energy_ratio > 0.50 or spec.low_energy_ratio > 0.20)
        and amp.crest_factor < 10.0
        and texture.temporal_entropy < 1.25
    )
    if is_engine_rumble or is_urban_traffic or ("engine" in category.lower() or "traffic" in category.lower()):
        has_traffic_or_engine = True
        detected_elements.append("Traffic & Urban Ambience")
        settings_parts.append("Harsh tire screech notched, horn honks auto-tuned to scale brass, sub-rumble sidechained")
        styles.extend(["cyberpunk", "lofi", "techno", "trap", "ambient"])

    # 2. Detect Human Speech / Vocal
    # Must distinguish genuine voice formants from mechanical rumble or environmental noise
    has_speech_or_vocal = False
    is_human_pitch_range = (75.0 <= pitch.fundamental_hz <= 450.0) if pitch.has_reliable_pitch else False
    has_speech_dynamics = (1.8 <= amp.crest_factor <= 15.0) and (texture.noisiness < 0.70)
    has_vocal_harmonicity = texture.harmonicity > 0.40 and pitch.pitch_confidence >= 0.45

    if category == "VOCAL":
        has_speech_or_vocal = True
    elif has_traffic_or_engine:
        # Traffic and environmental recordings must NEVER be treated as vocal speech
        has_speech_or_vocal = False
    else:
        # General audio takes: STRICT verification of human vocal pitch and harmonic formants
        if pitch.has_reliable_pitch and is_human_pitch_range and has_speech_dynamics and has_vocal_harmonicity:
            if spec.low_energy_ratio < 0.70:  # Allow natural male/female speech formants, exclude pure sub rumbles
                has_speech_or_vocal = True

    if has_speech_or_vocal:
        detected_elements.append("Voice & Speech")
        settings_parts.append("Scale Auto-Tuner active on speech syllables across all themes")
        styles.extend(["pop", "hiphop", "trap", "lofi", "rnb", "house"])

    # 3. Detect Vocal Humming / Whistling
    has_humming = False
    if pitch.has_reliable_pitch and pitch.pitch_confidence > 0.50 and not has_traffic_or_engine:
        if (texture.harmonicity > 0.45 or is_human_pitch_range) and texture.noisiness < 0.40 and rhythm.rhythmic_density < 3.0:
            has_humming = True
            detected_elements.append("Tuned Hum / Whistle")
            settings_parts.append("Pitch quantized to nearest scale notes with parallel 3rd/5th harmonization")
            styles.extend(["pop", "lofi", "ambient", "indie", "electronic"])

    # 4. Detect Vocal Beatbox / Mouth Percussion
    has_beatbox = False
    is_vocal_tract = (spec.mid_energy_ratio > 0.25) or (pitch.has_reliable_pitch and 70.0 <= pitch.fundamental_hz <= 450.0)
    has_mouth_dynamics = (amp.crest_factor > 2.8 and (rhythm.rhythmic_density >= 1.0 or len(rhythm.onset_samples) >= 4)) or len(rhythm.onset_samples) >= 8
    has_kick_and_snare_spectrum = spec.low_energy_ratio > 0.15 and (spec.high_energy_ratio > 0.04 or spec.spectral_centroid > 1400.0 or len(rhythm.onset_samples) >= 6)
    is_percussive_bursts = texture.temporal_entropy > 0.20 or rhythm.has_reliable_rhythm
    if is_vocal_tract and has_mouth_dynamics and has_kick_and_snare_spectrum and is_percussive_bursts and texture.harmonicity < 0.55:
        if not has_traffic_or_engine or (amp.crest_factor > 8.0 and texture.temporal_entropy > 1.2):
            has_beatbox = True
            detected_elements.append("Vocal Beatbox")
            settings_parts.append("Tri-band punch separation (Mouth Kick / Snare / Hats) with groove locking")
            styles.extend(["hiphop", "boom_bap", "trap", "drill", "funk", "electro"])

    # 5. Detect Percussive Foley / Taps / Clicks
    has_foley_percussive = False
    if (amp.crest_factor > 3.0 or rhythm.rhythmic_density > 0.8) and ("Vocal Beatbox" not in detected_elements):
        if not has_traffic_or_engine or rhythm.has_reliable_rhythm or amp.crest_factor > 8.5:
            has_foley_percussive = True
            detected_elements.append("Percussive Foley")
            settings_parts.append("Crisp transient isolation with stereo micro-panning")
            styles.extend(["minimal", "house", "electro", "hiphop"])

    # 6. Detect Ambient Bed / Atmosphere
    has_ambient_bed = False
    if amp.crest_factor < 3.0 or spec.spectral_flatness > 0.14 or rhythm.rhythmic_density < 0.6:
        has_ambient_bed = True
        if "Traffic & Urban Ambience" not in detected_elements:
            detected_elements.append("Ambient Bed")
            settings_parts.append("Stereo width expansion and breathing dynamic sidechain")
            styles.extend(["ambient", "downtempo", "chillout", "lofi"])

    # If no specific profile was detected, default based on standard category
    if not detected_elements:
        detected_elements.append(f"{category.title()} Audio")
        settings_parts.append("Adaptive resonance and multi-scale slice leveling applied")
        styles.extend(["pop", "lofi", "ambient", "house"])

    # Primary category determination
    if len(detected_elements) >= 2:
        primary_cat = "COMPOSITE"
        elem_short = [e.split()[0] for e in detected_elements[:3]]
        display_title = f"Composite Audio ({' + '.join(elem_short)})"
    elif has_traffic_or_engine:
        primary_cat = "TRAFFIC"
        display_title = "🚗 Traffic & Urban Street Audio"
    else:
        primary_cat = category
        display_title = detected_elements[0] if detected_elements else f"{category.title()} Audio"

    # Deduplicate styles and settings
    seen_styles = set()
    unique_styles = [s for s in styles if not (s in seen_styles or seen_styles.add(s))]
    summary_text = " • ".join(settings_parts)

    autotune_active = (has_speech_or_vocal or has_humming or (category in ["VOCAL", "MELODIC", "HARMONIC"])) and not has_traffic_or_engine

    return SmartSourceProfile(
        primary_category=primary_cat,
        display_title=display_title,
        has_speech_or_vocal=has_speech_or_vocal,
        has_humming=has_humming,
        has_beatbox=has_beatbox,
        has_traffic_or_engine=has_traffic_or_engine,
        has_foley_percussive=has_foley_percussive,
        has_ambient_bed=has_ambient_bed,
        autotune_enabled=autotune_active,
        autotune_strength=1.0 if (has_speech_or_vocal or has_humming) else 0.75,
        detected_elements=detected_elements,
        smart_settings_summary=summary_text,
        recommended_styles=unique_styles[:6]
    )


def analyze_audio(
    audio: np.ndarray,
    sr: int = 44100,
    on_progress: Optional[Callable[[int, str], None]] = None
) -> CompleteAnalysis:
    """Run full DSP feature extraction suite and rule-based classifier with continuous progress reporting."""
    if on_progress:
        on_progress(22, "Decoding acoustic waveform & dynamic range...")
    amp = compute_amplitude_features(audio, sr=sr)

    if on_progress:
        on_progress(28, "Extracting spectral centroid & frequency rolloff...")
    spec = analyze_spectrum(audio, sr=sr)

    if on_progress:
        on_progress(38, "Detecting rhythmic pulses & transient intervals...")
    rhythm = analyze_rhythm(audio, sr=sr)

    if on_progress:
        on_progress(45, "Estimating fundamental pitch & harmonic chroma...")
    pitch = analyze_pitch(audio, sr=sr)

    texture = compute_texture_features(audio, spec, rhythm, sr=sr)
    category, confidence, reasons = classify_source(amp, spec, rhythm, pitch, texture)

    smart_profile = detect_smart_source_profile(
        audio=audio,
        amp=amp,
        spec=spec,
        rhythm=rhythm,
        pitch=pitch,
        texture=texture,
        category=category,
        sr=sr
    )

    if on_progress:
        on_progress(50, f"Acoustic DNA analyzed: {smart_profile.display_title}!")

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
        sr=sr,
        smart_profile=smart_profile
    )
