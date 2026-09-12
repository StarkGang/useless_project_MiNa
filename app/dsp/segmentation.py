"""
Multi-Scale Source Audio Segmentation & Acoustic Role Classification
TheUnnecessaryFM Philosophy: The recording is the instrument.

Extracts micro (~20-400ms), medium (~0.5-3s), and chronological source slices
from across 0-100% of the ENTIRE uploaded recording.
Guarantees perceptual continuity via anti-click windowing, per-slice RMS leveling,
and robust fallbacks for quiet or low-transient inputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .rhythm import analyze_rhythm


@dataclass
class AudioSlice:
    audio: np.ndarray              # Mono float32 slice with anti-click fades and RMS leveling
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
    chronological_slices: List[AudioSlice] = field(default_factory=list)


def apply_fade(
    audio: np.ndarray,
    fade_samples: int = 64,
    fade_type: str = "cosine"
) -> np.ndarray:
    """
    Apply fade-in and fade-out to avoid discontinuities at boundaries.
    Guarantees that boundaries start and end strictly at or near 0 amplitude.
    Supports both 1D (mono) and 2D (channels, samples) numpy arrays.
    """
    if audio is None or len(audio) == 0:
        return audio
    out = audio.copy()
    n = out.shape[-1]
    if n <= 1:
        return out * 0.0
    f_len = min(n // 2, max(2, fade_samples))
    if f_len <= 1:
        return out

    if fade_type == "linear":
        fade_in = np.linspace(0.0, 1.0, f_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, f_len, dtype=np.float32)
    else:  # cosine / equal-power Hann
        t = np.linspace(0.0, np.pi, f_len, dtype=np.float32)
        fade_in = (0.5 - 0.5 * np.cos(t)).astype(np.float32)
        fade_out = (0.5 + 0.5 * np.cos(t)).astype(np.float32)

    if out.ndim == 1:
        out[:f_len] *= fade_in
        out[-f_len:] *= fade_out
    elif out.ndim == 2:
        out[:, :f_len] *= fade_in
        out[:, -f_len:] *= fade_out
    return out


def apply_slice_envelope(audio: np.ndarray, attack_ms: float = 3.0, release_ms: float = 8.0, sr: int = 44100) -> np.ndarray:
    """Applies smooth anti-click Hann fade-in and fade-out to prevent boundary discontinuities."""
    if len(audio) <= 4:
        return audio.copy()
    out = audio.copy()
    att_samples = min(len(out) // 4, max(2, int((attack_ms / 1000.0) * sr)))
    rel_samples = min(len(out) // 4, max(2, int((release_ms / 1000.0) * sr)))
    if att_samples > 1:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, att_samples, dtype=np.float32)))
        out[:att_samples] *= fade_in
    if rel_samples > 1:
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, rel_samples, dtype=np.float32)))
        out[-rel_samples:] *= fade_out
    return out


# Backward compatibility alias
_apply_fades = apply_slice_envelope


def normalize_slice_rms(audio: np.ndarray, target_rms: float = 0.12, max_gain: float = 6.0) -> np.ndarray:
    """
    Normalizes RMS of slice to a balanced target to prevent volume pumping when
    combining slices from disparate sections of a recording.
    """
    if len(audio) == 0:
        return audio
    cur_rms = float(np.sqrt(np.mean(audio ** 2)))
    if cur_rms < 1e-4:
        return audio.copy()
    gain = min(max_gain, max(0.2, target_rms / cur_rms))
    scaled = audio * gain
    # Soft saturation if peak exceeds 0.95
    pk = float(np.max(np.abs(scaled)))
    if pk > 0.95:
        scaled = np.tanh(scaled * 1.1) * 0.92
    return scaled.astype(np.float32)


def _analyze_slice(chunk: np.ndarray, sr: int = 44100) -> Tuple[float, float, float, float, float, float]:
    """Computes (rms, centroid, flatness, crest_factor, pitch_conf, dominant_freq)."""
    if len(chunk) < 64:
        return 0.0, 1000.0, 0.5, 1.0, 0.0, 0.0

    abs_c = np.abs(chunk)
    peak = float(np.max(abs_c))
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    crest = float(peak / max(rms, 1e-6))

    # Fast direct FFT spectral centroid and flatness
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

    # FFT-based normalized autocorrelation pitch confidence — O(N log N)
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
    spanning micro, medium, and long sound segments from 0% to 100% of the timeline.
    """
    total_len = len(audio)
    total_dur = total_len / sr
    if total_len == 0:
        empty_slice = AudioSlice(np.zeros(1024, dtype=np.float32), 0.0, 0.02, "TEXTURE", 0.0, 1000.0, 0.5, 1.0, 0.0, 0.0)
        return SourcePalette(
            all_slices=[empty_slice], impacts=[empty_slice], pulses=[empty_slice],
            movements=[empty_slice], textures=[empty_slice], drones=[empty_slice],
            ambience=[empty_slice], tonal=[empty_slice], accents=[empty_slice],
            source_duration=0.0, total_slices_extracted=1,
            chronological_slices=[empty_slice]
        )

    # 1. Onset Detection across the entire recording
    if onset_samples is None or len(onset_samples) == 0:
        try:
            rhythm_info = analyze_rhythm(audio, sr=sr)
            onset_samples = np.array(rhythm_info.onset_samples, dtype=int)
        except Exception:
            onset_samples = np.array([], dtype=int)

    all_slices: List[AudioSlice] = []

    # 2. Extract Micro Transient Sounds across 0-100% of timeline
    # If onsets are present, sample evenly across all onsets
    if len(onset_samples) >= 4:
        # Filter minimum refractory spacing (50ms)
        min_samp_dist = int(0.05 * sr)
        filtered_onsets = [onset_samples[0]]
        for o in onset_samples[1:]:
            if o - filtered_onsets[-1] >= min_samp_dist and o < total_len:
                filtered_onsets.append(o)

        num_micro = min(len(filtered_onsets), 24)
        chosen_indices = np.linspace(0, len(filtered_onsets) - 1, num_micro, dtype=int)
        for idx in chosen_indices:
            ons = filtered_onsets[idx]
            pre = int(0.005 * sr)  # 5ms pre-roll to catch transient attack peak
            dur_sec = 0.16 + 0.12 * ((idx % 3) / 2.0)  # 160ms to 280ms
            post = int(dur_sec * sr)
            s_start = max(0, ons - pre)
            s_end = min(total_len, s_start + post)
            if s_end - s_start > int(0.02 * sr):
                chunk = audio[s_start:s_end]
                chunk = apply_slice_envelope(chunk, attack_ms=2.0, release_ms=10.0, sr=sr)
                chunk = normalize_slice_rms(chunk, target_rms=0.14)
                rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(chunk, sr=sr)

                if p_conf > 0.60:
                    role = "TONAL"
                elif cent > 2800 or flat > 0.45:
                    role = "PULSE"
                elif crest > 2.2:
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
    else:
        # Fallback for low-onset or quiet recordings: Uniform grid chopping
        num_grid = min(16, max(6, int(total_dur * 2)))
        grid_starts = np.linspace(0, max(0, total_len - int(0.25 * sr)), num_grid, dtype=int)
        for s_start in grid_starts:
            s_len = min(int(0.24 * sr), total_len - s_start)
            if s_len >= int(0.04 * sr):
                chunk = audio[s_start:s_start + s_len]
                chunk = apply_slice_envelope(chunk, attack_ms=2.0, release_ms=10.0, sr=sr)
                chunk = normalize_slice_rms(chunk, target_rms=0.12)
                rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(chunk, sr=sr)
                role = "IMPACT" if crest > 2.0 else ("PULSE" if cent > 2200 else "TEXTURE")
                all_slices.append(AudioSlice(
                    audio=chunk,
                    start_sec=round(s_start / sr, 3),
                    duration=round(s_len / sr, 3),
                    role=role,
                    rms=rms,
                    centroid=cent,
                    flatness=flat,
                    crest_factor=crest,
                    pitch_conf=p_conf,
                    dominant_pitch_hz=dom_f
                ))

    # 3. Extract Medium Sounds (0.4s - 2.5s) distributed across 0-100% of the timeline
    num_medium = 16
    med_dur = min(max(0.45, total_dur * 0.12), 2.2)
    s_len_med = int(med_dur * sr)

    for i in range(num_medium):
        fraction = (i + 0.5) / num_medium
        target_center = int(fraction * total_len)
        s_start = max(0, target_center - s_len_med // 2)
        s_end = min(total_len, s_start + s_len_med)
        if s_end - s_start >= int(0.3 * sr):
            chunk = audio[s_start:s_end]
            chunk = apply_slice_envelope(chunk, attack_ms=12.0, release_ms=20.0, sr=sr)
            chunk = normalize_slice_rms(chunk, target_rms=0.12)
            rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(chunk, sr=sr)

            if p_conf > 0.55:
                role = "TONAL"
            elif crest > 3.0 and rms > 0.04:
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

    # 4. Extract Chronological Timeline Slices for Continuous Atmospheric Bed
    # Traverses the full recording sequentially so that speech, environmental noise,
    # or musical movement evolves naturally from beginning to end
    chronological_slices: List[AudioSlice] = []
    num_chrono = 8
    chrono_dur = min(max(1.2, total_dur / float(num_chrono) * 1.5), 8.0)
    chrono_len = min(total_len, int(chrono_dur * sr))

    if total_len > chrono_len:
        chrono_starts = np.linspace(0, total_len - chrono_len, num_chrono, dtype=int)
        for s_start in chrono_starts:
            s_end = s_start + chrono_len
            chunk = audio[s_start:s_end]
            chunk = apply_slice_envelope(chunk, attack_ms=30.0, release_ms=45.0, sr=sr)
            chunk = normalize_slice_rms(chunk, target_rms=0.11)
            rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(chunk, sr=sr)
            role = "DRONE" if (p_conf > 0.50 or cent < 500) else "AMBIENCE"

            sl = AudioSlice(
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
            )
            all_slices.append(sl)
            chronological_slices.append(sl)
    else:
        # Full recording used as ambience
        chunk = audio.copy()
        chunk = apply_slice_envelope(chunk, attack_ms=30.0, release_ms=45.0, sr=sr)
        chunk = normalize_slice_rms(chunk, target_rms=0.11)
        rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(chunk, sr=sr)
        sl = AudioSlice(
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
        )
        all_slices.append(sl)
        chronological_slices.append(sl)

    # 5. Partition into distinct acoustic role buckets
    impacts = [s for s in all_slices if s.role == "IMPACT" or (s.duration < 0.45 and s.crest_factor > 2.0 and s.centroid < 2600)]
    pulses = [s for s in all_slices if s.role == "PULSE" or (s.duration < 0.45 and (s.centroid >= 2400 or s.flatness > 0.4))]
    movements = [s for s in all_slices if s.role == "MOVEMENT" or (0.35 <= s.duration <= 4.5 and s.crest_factor <= 3.2)]
    textures = [s for s in all_slices if s.role == "TEXTURE"]
    drones = [s for s in all_slices if s.role == "DRONE" or (s.duration >= 1.5 and s.centroid < 750)]
    ambience = [s for s in all_slices if s.role == "AMBIENCE" or s.duration >= 1.5]
    tonal = [s for s in all_slices if s.pitch_conf >= 0.45 or s.role == "TONAL"]
    accents = [s for s in all_slices if s.role == "ACCENT" or s.crest_factor > 3.2]

    # Robust fallbacks to guarantee every bucket has at least 1 usable slice
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
    if not tonal:
        tonal = movements or [fallback_slice]
    if not chronological_slices:
        chronological_slices = ambience or [fallback_slice]

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
        total_slices_extracted=len(all_slices),
        chronological_slices=chronological_slices
    )
