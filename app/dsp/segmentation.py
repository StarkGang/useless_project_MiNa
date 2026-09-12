"""
Multi-Scale Source Audio Segmentation & Acoustic Role Classification
TheUnnecessaryFM Philosophy: The recording is the instrument.

Extracts micro (~20-400ms), medium (~0.5-3s), and chronological source slices
from across 0-100% of the ENTIRE uploaded recording.
Guarantees perceptual continuity via anti-click windowing, per-slice RMS leveling,
and robust fallbacks for quiet or low-transient inputs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal

from .rhythm import analyze_rhythm


@dataclass
class AudioSlice:
    audio: np.ndarray              # Mono float32 slice with anti-click fades and RMS leveling
    start_sec: float               # Offset in original recording
    duration: float                # Duration in seconds
    role: str                      # IMPACT, PULSE, MOVEMENT, TEXTURE, DRONE, TONAL, ACCENT, AMBIENCE, VOCAL
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
    vocal_chops: List[AudioSlice] = field(default_factory=list)  # Clean, recognizable syllabic voice/sound chops
    full_vocal_phrases: List[AudioSlice] = field(default_factory=list) # Intact continuous singing/speech phrases (1.2s - 8s)
    lead_vocal_take: Optional[np.ndarray] = None # Entire dehissed vocal recording for lead vocal mode
    beatbox_kicks: List[AudioSlice] = field(default_factory=list) # Punchy low-end mouth kicks / lip bass
    beatbox_snares: List[AudioSlice] = field(default_factory=list) # Snappy mid-frequency mouth snares / claps / clicks
    beatbox_hats: List[AudioSlice] = field(default_factory=list) # Crisp high-frequency mouth hi-hats / ticks / sizzle
    traffic_horns: List[AudioSlice] = field(default_factory=list) # Car horns / siren tonal bursts
    humming_slices: List[AudioSlice] = field(default_factory=list) # Sustained tonal vocal hums / whistles


def dehiss_audio(
    audio: np.ndarray,
    sr: int = 44100,
    high_cut_hz: float = 14500.0,
    gate_threshold_db: float = -46.0
) -> np.ndarray:
    """
    Studio-grade vocal cleanup:
    1. High-pass filter at 50Hz (removes low-end mic thumps and room rumble)
    2. Gentle ultrasonic roll-off above 14.5kHz (removes preamp digital hiss while preserving vocal air)
    3. Smooth studio downward expander (attenuates quiet noise floor without modulating voice waveforms or crackling)
    """
    if audio is None or len(audio) < 16:
        return audio

    out = audio.astype(np.float32).copy()

    # 1. Gentle high-pass filter at 50 Hz to eliminate DC and rumble
    try:
        b_hp, a_hp = signal.butter(2, max(20.0, 50.0) / (sr * 0.5), btype='high')
        out = signal.lfilter(b_hp, a_hp, out)
    except Exception:
        pass

    # 2. Gentle ultrasonic roll-off at high_cut_hz
    try:
        nyq = sr * 0.5
        cut = min(high_cut_hz, nyq * 0.92)
        if cut < nyq * 0.90:
            b_lp, a_lp = signal.butter(2, cut / nyq, btype='low')
            out = signal.lfilter(b_lp, a_lp, out)
    except Exception:
        pass

    # 3. Smooth studio downward expander (50ms window + lowpass envelope smoothing)
    pk = float(np.max(np.abs(out)))
    if pk > 1e-5:
        win_size = max(64, int(0.050 * sr))  # 50ms window avoids audio-rate wave modulation
        kernel = np.ones(win_size, dtype=np.float32) / win_size
        power_env = np.convolve(out ** 2, kernel, mode='same')
        rms_env = np.sqrt(np.maximum(power_env, 1e-9))
        thresh = pk * (10.0 ** (gate_threshold_db / 20.0))

        # Soft expansion curve with minimum gain floor at 0.12 (-18dB) to prevent abrupt chatter
        raw_gain = np.clip((rms_env - thresh * 0.3) / max(thresh * 0.7, 1e-6), 0.12, 1.0)
        # Smooth gain curve with 20Hz lowpass filter to guarantee zero amplitude-modulation distortion
        try:
            b_g, a_g = signal.butter(1, min(0.40, 20.0 / (sr * 0.5)), btype='low')
            smooth_gain = signal.lfilter(b_g, a_g, raw_gain)
            smooth_gain = np.clip(smooth_gain, 0.12, 1.0).astype(np.float32)
            out = out * smooth_gain
        except Exception:
            out = out * raw_gain

    return out.astype(np.float32)



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
    target_slice_count: int = 48,
    smart_profile: Optional[Any] = None
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

    # 4.4. Extract Continuous Singing / Vocal Phrases (1.2s - 8.0s) & Full Lead Take
    # Preserves full lyrics, intact words, and vocal melodies without truncating mid-word
    full_vocal_phrases: List[AudioSlice] = []
    lead_vocal_take: Optional[np.ndarray] = None
    vocal_chops: List[AudioSlice] = []

    is_vocal_source = bool(
        smart_profile and (
            getattr(smart_profile, 'has_speech_or_vocal', False)
            or getattr(smart_profile, 'has_humming', False)
            or getattr(smart_profile, 'primary_category', '') == "VOCAL"
        )
    )

    if is_vocal_source:
        clean_lead_audio = dehiss_audio(audio, sr=sr, high_cut_hz=6800.0, gate_threshold_db=-38.0)
        lead_vocal_take = clean_lead_audio.copy()

        # Short-time RMS energy envelope (60ms hop) to detect natural breath/pause points
        hop_vad = int(0.060 * sr)
        win_vad = int(0.120 * sr)
        if total_len > win_vad:
            n_vad = max(1, (total_len - win_vad) // hop_vad + 1)
            vad_energy = np.zeros(n_vad, dtype=np.float32)
            for i in range(n_vad):
                c = audio[i * hop_vad : i * hop_vad + win_vad]
                vad_energy[i] = float(np.sqrt(np.mean(c ** 2))) + 1e-8

            # Pause threshold: 25% of median active vocal energy
            active_e = vad_energy[vad_energy > 0.005]
            med_e = float(np.median(active_e)) if len(active_e) > 0 else 0.02
            pause_thresh = max(0.006, med_e * 0.28)
            is_speech = vad_energy > pause_thresh

            # Identify continuous speech spans
            diff_s = np.diff(is_speech.astype(int))
            p_starts = list(np.where(diff_s == 1)[0] + 1)
            p_ends = list(np.where(diff_s == -1)[0] + 1)
            if is_speech[0]:
                p_starts.insert(0, 0)
            if is_speech[-1]:
                p_ends.append(n_vad)

            pad_samp = int(0.10 * sr)
            min_phrase_len = int(1.2 * sr)
            max_phrase_len = int(8.0 * sr)

            for st_f, en_f in zip(p_starts, p_ends):
                s_s = max(0, st_f * hop_vad - pad_samp)
                e_s = min(total_len, en_f * hop_vad + win_vad + pad_samp)
                if (e_s - s_s) >= min_phrase_len:
                    if (e_s - s_s) > max_phrase_len:
                        e_s = s_s + max_phrase_len
                    phrase_chunk = clean_lead_audio[s_s:e_s].copy()
                    fade_p = min(int(0.015 * sr), len(phrase_chunk) // 8)
                    if fade_p > 1:
                        w_in = np.linspace(0.0, 1.0, fade_p, dtype=np.float32)
                        phrase_chunk[:fade_p] *= w_in
                        phrase_chunk[-fade_p:] *= w_in[::-1]
                    p_rms, p_cent, p_flat, p_crest, p_conf, p_dom = _analyze_slice(phrase_chunk, sr=sr)
                    full_vocal_phrases.append(
                        AudioSlice(
                            audio=phrase_chunk,
                            start_sec=round(s_s / sr, 3),
                            duration=round((e_s - s_s) / sr, 3),
                            role="VOCAL_PHRASE",
                            rms=p_rms,
                            centroid=p_cent,
                            flatness=p_flat,
                            crest_factor=p_crest,
                            pitch_conf=p_conf,
                            dominant_pitch_hz=p_dom
                        )
                    )

        # Fallback: If no multi-pause segmentation occurred, the whole take is Phrase 1
        if not full_vocal_phrases and total_len >= int(1.0 * sr):
            t_fade = min(int(0.02 * sr), total_len // 8)
            whole_chunk = clean_lead_audio.copy()
            if t_fade > 1:
                w_t = np.linspace(0.0, 1.0, t_fade, dtype=np.float32)
                whole_chunk[:t_fade] *= w_t
                whole_chunk[-t_fade:] *= w_t[::-1]
            p_rms, p_cent, p_flat, p_crest, p_conf, p_dom = _analyze_slice(whole_chunk, sr=sr)
            full_vocal_phrases.append(
                AudioSlice(
                    audio=whole_chunk,
                    start_sec=0.0,
                    duration=round(total_len / sr, 3),
                    role="VOCAL_PHRASE",
                    rms=p_rms,
                    centroid=p_cent,
                    flatness=p_flat,
                    crest_factor=p_crest,
                    pitch_conf=p_conf,
                    dominant_pitch_hz=p_dom
                )
            )

        # 4.5. Extract Syllable-Aware Vocal / Voice Chops (180ms - 420ms)
        # Focuses on voiced phonemes, distinct words, and articulate acoustic gestures
        chop_durs = [0.22, 0.28, 0.35, 0.42]

        # Candidate points: use onsets if available, otherwise spaced energy points
        if len(onset_samples) >= 3:
            vocal_starts = [int(ons) for ons in onset_samples if ons < total_len - int(0.20 * sr)]
        else:
            vocal_starts = list(np.linspace(0, max(0, total_len - int(0.35 * sr)), 12, dtype=int))

        # Downsample candidate count to at most 16
        if len(vocal_starts) > 16:
            step_v = len(vocal_starts) / 16.0
            vocal_starts = [vocal_starts[int(i * step_v)] for i in range(16)]

        for idx, v_start in enumerate(vocal_starts):
            c_dur = chop_durs[idx % len(chop_durs)]
            c_samples = int(c_dur * sr)
            v_end = min(total_len, v_start + c_samples)
            if v_end - v_start >= int(0.12 * sr):
                raw_chunk = audio[v_start:v_end].copy()
                # Clean hiss and room rumble while preserving speech formants
                clean_chunk = dehiss_audio(raw_chunk, sr=sr, high_cut_hz=6400.0, gate_threshold_db=-38.0)
                clean_chunk = apply_slice_envelope(clean_chunk, attack_ms=4.0, release_ms=16.0, sr=sr)
                clean_chunk = normalize_slice_rms(clean_chunk, target_rms=0.15)
                rms, cent, flat, crest, p_conf, dom_f = _analyze_slice(clean_chunk, sr=sr)

                # Filter for articulate speech/timbre (not purely flat static noise)
                if rms > 0.02 and flat < 0.70:
                    v_slice = AudioSlice(
                        audio=clean_chunk,
                        start_sec=round(v_start / sr, 3),
                        duration=round((v_end - v_start) / sr, 3),
                        role="VOCAL",
                        rms=rms,
                        centroid=cent,
                        flatness=flat,
                        crest_factor=crest,
                        pitch_conf=p_conf,
                        dominant_pitch_hz=dom_f
                    )
                    vocal_chops.append(v_slice)
                    all_slices.append(v_slice)

    # 5. Partition into distinct acoustic role buckets
    impacts = [s for s in all_slices if s.role == "IMPACT" or (s.duration < 0.45 and s.crest_factor > 2.0 and s.centroid < 2600)]
    pulses = [s for s in all_slices if s.role == "PULSE" or (s.duration < 0.45 and (s.centroid >= 2400 or s.flatness > 0.4))]
    movements = [s for s in all_slices if s.role == "MOVEMENT" or (0.35 <= s.duration <= 4.5 and s.crest_factor <= 3.2)]
    textures = [s for s in all_slices if s.role == "TEXTURE"]
    drones = [s for s in all_slices if s.role == "DRONE" or (s.duration >= 1.5 and s.centroid < 750)]
    ambience = [s for s in all_slices if s.role == "AMBIENCE" or s.duration >= 1.5]
    tonal = [s for s in all_slices if s.pitch_conf >= 0.45 or s.role == "TONAL"]
    accents = [s for s in all_slices if s.role == "ACCENT" or s.crest_factor > 3.2]

    # 5.5. Specialized Smart Sub-Role Buckets (for Beatbox, Humming, and Traffic)
    # Beatbox Kicks: High crest factor and low centroid (< 950 Hz)
    beatbox_kicks = [s for s in all_slices if s.crest_factor > 2.2 and s.centroid < 950 and s.duration < 0.50]
    # Beatbox Snares: Sharp transient in mid frequency (1000 - 4200 Hz)
    beatbox_snares = [s for s in all_slices if s.crest_factor > 2.4 and 1000 <= s.centroid <= 4200 and s.duration < 0.40]
    # Beatbox Hats: High centroid or high flatness with short duration
    beatbox_hats = [s for s in all_slices if (s.centroid > 3800 or s.flatness > 0.42) and s.duration < 0.22]

    # Humming Slices: High pitch confidence, sustained, low flatness
    humming_slices = [s for s in all_slices if s.pitch_conf >= 0.50 and s.flatness < 0.38 and s.duration >= 0.20]

    # Traffic Horns & Sirens: Tonal resonant bursts in 280 - 2600 Hz
    traffic_horns = [s for s in all_slices if (280.0 <= s.dominant_pitch_hz <= 2600.0 or s.pitch_conf > 0.50) and s.crest_factor > 2.0]

    # If traffic noise detected, apply gentle notch filter on ambient slices to remove harsh tire screech (> 3.2kHz)
    if smart_profile and getattr(smart_profile, 'has_traffic_or_engine', False):
        try:
            nyq = sr * 0.5
            b_notch, a_notch = signal.butter(1, min(0.90, 3200.0 / nyq), btype='low')
            for amb_sl in ambience:
                amb_sl.audio = signal.lfilter(b_notch, a_notch, amb_sl.audio).astype(np.float32)
        except Exception:
            pass

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
    if not vocal_chops and is_vocal_source:
        # Fallback to dehissed tonal and movement chunks only for genuine vocal sources
        vocal_chops = [
            AudioSlice(
                audio=dehiss_audio(s.audio, sr=sr),
                start_sec=s.start_sec,
                duration=s.duration,
                role="VOCAL",
                rms=s.rms,
                centroid=s.centroid,
                flatness=s.flatness,
                crest_factor=s.crest_factor,
                pitch_conf=s.pitch_conf,
                dominant_pitch_hz=s.dominant_pitch_hz
            )
            for s in (tonal + movements)[:6]
        ] or [fallback_slice]

    if not beatbox_kicks:
        beatbox_kicks = impacts[:3] or [fallback_slice]
    if not beatbox_snares:
        beatbox_snares = (impacts[1:4] + pulses[:2]) or [fallback_slice]
    if not beatbox_hats:
        beatbox_hats = pulses[:4] or [fallback_slice]
    if not traffic_horns:
        traffic_horns = accents[:2] or [fallback_slice]
    if not humming_slices:
        humming_slices = tonal[:3] or [fallback_slice]

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
        chronological_slices=chronological_slices,
        vocal_chops=vocal_chops,
        full_vocal_phrases=full_vocal_phrases,
        lead_vocal_take=lead_vocal_take,
        beatbox_kicks=beatbox_kicks,
        beatbox_snares=beatbox_snares,
        beatbox_hats=beatbox_hats,
        traffic_horns=traffic_horns,
        humming_slices=humming_slices
    )
