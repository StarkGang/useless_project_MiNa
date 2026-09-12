"""
Multi-Scale Source Audio Segmentation & Acoustic Role Classification
TheUnnecessaryFM Philosophy: The recording is the instrument.

Extracts micro (~20-400ms), medium (~0.5-4s), and long (~4-20s) source slices
from across the ENTIRE uploaded recording.
Classifies slices into acoustic roles:
  IMPACT, PULSE, MOVEMENT, TEXTURE, DRONE, TONAL, ACCENT, AMBIENCE.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .rhythm import analyze_rhythm


@dataclass
class AudioSlice:
    audio: np.ndarray              # Mono float32 slice with anti-click fades
    start_sec: float               # Offset in original recording
    duration: float                # Duration in seconds
    role: str                      # IMPACT, PULSE, MOVEMENT, TEXTURE, DRONE, TONAL, ACCENT, AMBIENCE
    rms: float
    centroid: float                # Spectral centroid in Hz
    flatness: float                # Spectral flatness [0.0, 1.0]
    crest_factor: float            # Peak / RMS
    pitch_conf: float              # Pitch confidence [0.0, 1.0]
    dominant_pitch_hz: float       # If tonal, estimated frequency


@dataclass
class SourcePalette:
    all_slices: List[AudioSlice]
    impacts: List[AudioSlice]       # Low/mid percussive strikes, knocks, heavy transients
    pulses: List[AudioSlice]        # Crisp high-frequency clicks, ticks, taps
    movements: List[AudioSlice]     # Medium dynamic gestures, footsteps, sweeps, mechanical events
    textures: List[AudioSlice]      # Evolving granular/environmental textures
    drones: List[AudioSlice]        # Sustained low/mid resonant beds, engine hums, AC
    ambience: List[AudioSlice]      # Long foundational atmospheric beds (rain, room tone, wind)
    tonal: List[AudioSlice]         # Any slices with high pitch confidence
    accents: List[AudioSlice]       # Rare dramatic impacts/sweeps for structural markers
    source_duration: float
    total_slices_extracted: int


def _apply_fades(audio: np.ndarray, attack_ms: float = 3.0, release_ms: float = 8.0, sr: int = 44100) -> np.ndarray:
    """Applies smooth anti-click Hann fade-in and fade-out."""
    out = audio.copy()
    att_samples = min(len(out) // 4, int((attack_ms / 1000.0) * sr))
    rel_samples = min(len(out) // 4, int((release_ms / 1000.0) * sr))
    if att_samples > 1:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, att_samples)))
        out[:att_samples] *= fade_in
    if rel_samples > 1:
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, rel_samples)))
        out[-rel_samples:] *= fade_out
    return out


def _analyze_slice(chunk: np.ndarray, sr: int = 44100) -> Tuple[float, float, float, float, float, float]:
    """
    Computes (rms, centroid, flatness, crest_factor, pitch_conf, dominant_freq).
    """
    if len(chunk) < 64:
        return 0.0, 1000.0, 0.5, 1.0, 0.0, 0.0

    abs_c = np.abs(chunk)
    peak = float(np.max(abs_c))
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    crest = float(peak / max(rms, 1e-6))

    # Fast direct FFT spectral centroid and flatness (avoids 100+ librosa wrapper calls)
    n_fft = min(2048, 2 ** int(np.floor(np.log2(len(chunk)))))
    if n_fft >= 128:
        win = np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(chunk[:n_fft] * win))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        spec_sum = float(np.sum(spec))
        if spec_sum > 1e-7:
            cent = float(np.sum(freqs * spec) / spec_sum)
            power = spec ** 2 + 1e-12
            geom_mean = float(np.exp(np.mean(np.log(power))))
            arith_mean = float(np.mean(power))
            flat = float(np.clip(geom_mean / max(arith_mean, 1e-12), 0.0, 1.0))
        else:
            cent = 1000.0
            flat = 0.5
    else:
        cent = 1000.0
        flat = 0.5

    # FFT-based normalized autocorrelation pitch confidence — O(N log N) not O(N²)
    # Called ~34x per job so this fix is critical for slow containers
    pitch_conf = 0.0
    dom_freq = 0.0
    if len(chunk) >= 512:
        try:
            ac_chunk = chunk[:min(len(chunk), 4096)]
            n = len(ac_chunk)
            n_fft2 = 2 ** int(np.ceil(np.log2(2 * n - 1)))
            fx = np.fft.rfft(ac_chunk, n=n_fft2)
            corr = np.fft.irfft(fx * np.conj(fx))[:n]
            if corr[0] > 1e-8:
                corr_norm = corr / corr[0]
                min_lag = int(sr / 1200.0)
                max_lag = int(sr / 50.0)
                if max_lag < len(corr_norm):
                    search_zone = corr_norm[min_lag:max_lag]
                    best_lag_rel = np.argmax(search_zone)
                    peak_val = float(search_zone[best_lag_rel])
                    if peak_val > 0.35:
                        pitch_conf = float(np.clip((peak_val - 0.35) / 0.65, 0.0, 1.0))
                        best_lag = min_lag + best_lag_rel
                        dom_freq = float(sr / best_lag)
        except Exception:
            pass

    return rms, cent, flat, crest, pitch_conf, dom_freq


def build_source_palette(
    audio: np.ndarray,
    sr: int = 44100,
    onset_samples: Optional[np.ndarray] = None,
    target_slice_count: int = 48
) -> SourcePalette:
    """
    Analyzes the entire uploaded audio and extracts a rich multi-scale palette
    spanning micro, medium, and long sound segments.
    """
    total_len = len(audio)
    total_dur = total_len / sr
    if total_len == 0:
        empty_slice = AudioSlice(np.zeros(1024, dtype=np.float32), 0.0, 0.02, "TEXTURE", 0.0, 1000.0, 0.5, 1.0, 0.0, 0.0)
        return SourcePalette(
            all_slices=[empty_slice], impacts=[empty_slice], pulses=[empty_slice],
            movements=[empty_slice], textures=[empty_slice], drones=[empty_slice],
            ambience=[empty_slice], tonal=[empty_slice], accents=[empty_slice],
            source_duration=0.0, total_slices_extracted=1
        )

    # 1. Onset Detection across the entire recording (via pure NumPy spectral flux)
    if onset_samples is None or len(onset_samples) == 0:
        try:
            rhythm_info = analyze_rhythm(audio, sr=sr)
            onset_samples = np.array(rhythm_info.onset_samples, dtype=int)
        except Exception:
            onset_samples = np.array([], dtype=int)

    all_slices: List[AudioSlice] = []

    # 2. Extract Micro Sounds (20ms - 350ms): Clicks, transients, percussive taps
    # We select up to 24 diverse onsets across the timeline
    if len(onset_samples) > 0:
        # Sample evenly across the onsets
        num_micro = min(len(onset_samples), 12)
        indices = np.linspace(0, len(onset_samples) - 1, num_micro, dtype=int)
        for idx in indices:
            ons = onset_samples[idx]
            pre = int(0.005 * sr) # 5ms pre-roll
            dur_sec = float(np.random.uniform(0.06, 0.32))
            post = int(dur_sec * sr)
            s_start = max(0, ons - pre)
            s_end = min(total_len, s_start + post)
            if s_end - s_start > int(0.02 * sr):
                chunk = _apply_fades(audio[s_start:s_end], attack_ms=2.0, release_ms=12.0, sr=sr)
                rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(chunk, sr=sr)
                
                # Role assignment
                if p_conf > 0.60:
                    role = "TONAL"
                elif cent > 2800 or flat > 0.45:
                    role = "PULSE"
                elif crest > 2.5:
                    role = "IMPACT"
                else:
                    role = "TEXTURE"

                all_slices.append(AudioSlice(
                    audio=chunk,
                    start_sec=round(s_start / sr, 3),
                    duration=round((s_end - s_start) / sr, 3),
                    role=role,
                    rms=rms,
                    centroid=cent,
                    flatness=flat,
                    crest_factor=crest,
                    pitch_conf=p_conf,
                    dominant_pitch_hz=dom_f
                ))

    # 3. Extract Medium Sounds (0.5s - 4.0s): Gestures, footsteps, mechanical sweeps
    # Sample 12-16 medium segments distributed across timeline
    num_medium = 16
    med_durations = [0.6, 1.0, 1.5, 2.2, 3.2]
    for i in range(num_medium):
        fraction = (i + 0.5) / num_medium
        target_center = int(fraction * total_len)
        dur = med_durations[i % len(med_durations)]
        s_len = int(dur * sr)
        s_start = max(0, target_center - s_len // 2)
        s_end = min(total_len, s_start + s_len)
        if s_end - s_start >= int(0.4 * sr):
            chunk = _apply_fades(audio[s_start:s_end], attack_ms=15.0, release_ms=25.0, sr=sr)
            rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(chunk, sr=sr)

            # Role classification for medium segments
            if p_conf > 0.55:
                role = "TONAL"
            elif crest > 3.2 and rms > 0.04:
                role = "ACCENT"
            elif flat > 0.35 and cent > 1500:
                role = "TEXTURE"
            else:
                role = "MOVEMENT"

            all_slices.append(AudioSlice(
                audio=chunk,
                start_sec=round(s_start / sr, 3),
                duration=round((s_end - s_start) / sr, 3),
                role=role,
                rms=rms,
                centroid=cent,
                flatness=flat,
                crest_factor=crest,
                pitch_conf=p_conf,
                dominant_pitch_hz=dom_f
            ))

    # 4. Extract Long Sounds (4.0s - 16.0s): Atmospheric beds, environmental drones
    # Sample 4-6 large sustained sections across beginning, middle, and end
    num_long = 6
    long_dur = min(total_dur * 0.45, 12.0)
    long_dur = max(3.0, long_dur)
    long_len = int(long_dur * sr)

    if total_len > long_len:
        long_starts = np.linspace(0, total_len - long_len, num_long, dtype=int)
        for s_start in long_starts:
            s_end = s_start + long_len
            chunk = _apply_fades(audio[s_start:s_end], attack_ms=40.0, release_ms=60.0, sr=sr)
            rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(chunk, sr=sr)

            if p_conf > 0.50 or cent < 450:
                role = "DRONE"
            else:
                role = "AMBIENCE"

            all_slices.append(AudioSlice(
                audio=chunk,
                start_sec=round(s_start / sr, 3),
                duration=round((s_end - s_start) / sr, 3),
                role=role,
                rms=rms,
                centroid=cent,
                flatness=flat,
                crest_factor=crest,
                pitch_conf=p_conf,
                dominant_pitch_hz=dom_f
            ))
    else:
        # Full recording used as ambience
        chunk = _apply_fades(audio, attack_ms=40.0, release_ms=60.0, sr=sr)
        rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(chunk, sr=sr)
        all_slices.append(AudioSlice(
            audio=chunk,
            start_sec=0.0,
            duration=round(total_dur, 3),
            role="AMBIENCE",
            rms=rms,
            centroid=cent,
            flatness=flat,
            crest_factor=crest,
            pitch_conf=p_conf,
            dominant_pitch_hz=dom_f
        ))

    # 5. Partition into distinct acoustic role buckets
    impacts = [s for s in all_slices if s.role == "IMPACT" or (s.duration < 0.45 and s.crest_factor > 2.0 and s.centroid < 2500)]
    pulses = [s for s in all_slices if s.role == "PULSE" or (s.duration < 0.45 and (s.centroid >= 2500 or s.flatness > 0.4))]
    movements = [s for s in all_slices if s.role == "MOVEMENT" or (0.4 <= s.duration <= 4.5 and s.crest_factor <= 3.0)]
    textures = [s for s in all_slices if s.role == "TEXTURE"]
    drones = [s for s in all_slices if s.role == "DRONE" or (s.duration >= 3.0 and s.centroid < 700)]
    ambience = [s for s in all_slices if s.role == "AMBIENCE" or s.duration >= 3.0]
    tonal = [s for s in all_slices if s.pitch_conf >= 0.45 or s.role == "TONAL"]
    accents = [s for s in all_slices if s.role == "ACCENT" or s.crest_factor > 3.5]

    # Fallbacks to guarantee every bucket has at least 1 usable slice
    fallback_slice = all_slices[0]
    if not impacts:
        impacts = [s for s in all_slices if s.duration < 0.8] or [fallback_slice]
    if not pulses:
        pulses = [s for s in all_slices if s.duration < 0.5] or [fallback_slice]
    if not movements:
        movements = [s for s in all_slices if 0.3 <= s.duration <= 5.0] or [fallback_slice]
    if not textures:
        textures = movements or [fallback_slice]
    if not drones:
        drones = ambience or [fallback_slice]
    if not ambience:
        ambience = drones or [fallback_slice]
    if not accents:
        accents = impacts or [fallback_slice]

    return SourcePalette(
        all_slices=all_slices,
        impacts=impacts,
        pulses=pulses,
        movements=movements,
        textures=textures,
        drones=drones,
        ambience=ambience,
        tonal=tonal,
        accents=accents,
        source_duration=round(total_dur, 2),
        total_slices_extracted=len(all_slices)
    )
