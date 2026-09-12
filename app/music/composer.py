"""
Master Procedural Composer for TheUnnecessaryFM
Philosophy: THE NOISE IS THE INSTRUMENT, PRODUCED INTO REAL MUSIC.

Turns ANY audio recording into a punchy, enjoyable, head-nodding ~60s musical composition:
  1. Drums: Kick, snare, hi-hats, and foley chops forged from source transients.
  2. Bass: Thick, analog-saturated sub-bass fused with source low-end texture.
  3. Chords: Lush harmonic chord progressions resonated from the source audio.
  4. Melody: Memorable melodic hook played via tuned physical resonant modeling.
  5. Atmosphere: Continuous ambient recording bed with subtle sidechain ducking.
"""

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal

from ..dsp.analysis import CompleteAnalysis
from ..dsp.effects import (
    apply_panning,
    master_audio,
    ping_pong_delay,
    schroeder_reverb,
    soft_saturation
)
from ..dsp.filters import design_biquad_lowpass, apply_filter, resonant_filter_bank
from ..dsp.granular import create_granular_pad
from ..dsp.pitch import midi_to_hz
from ..dsp.preprocess import PreprocessedAudio
from ..dsp.segmentation import (
    AudioSlice,
    SourcePalette,
    build_source_palette,
    apply_slice_envelope,
    normalize_slice_rms,
    apply_fade
)
from ..dsp.synthesis import (
    render_karplus_strong_noise_note,
    render_noise_bass_note,
    render_trap_808_glide_bass,
    render_noise_downlifter,
    render_noise_hihat,
    render_noise_instrument_note,
    render_noise_kick,
    render_noise_open_hihat,
    render_noise_riser,
    render_noise_snare
)
from ..utils.random import SeededRNG, generate_seed
from .artist_profiles import ArtistProfile, get_random_artist_for_genre, get_artist_by_id
from .arrangement import CompositionArrangement, Section, plan_arrangement
from .chords import select_chord_progression
from .melody import sequence_melody
from .rhythm_generator import generate_rhythm
from .scales import MusicalScale, select_scale
from .scoring import CandidateScore, score_composition


class BagSelector:
    """
    Round-robin shuffled bag selector with guaranteed no immediate repeats.
    Ensures varied selection without unnatural rapid looping or modulo repetition.
    """
    def __init__(self, items: List[Any], rng: SeededRNG):
        self.items = [x for x in items if x is not None] if items else []
        self.rng = rng
        self.pool: List[Any] = []
        self.last_item: Any = None
        self._refill()

    def _refill(self):
        if not self.items:
            return
        shuffled = self.rng.shuffle(self.items)
        if len(shuffled) > 1 and self.last_item is not None and shuffled[0] is self.last_item:
            shuffled[0], shuffled[-1] = shuffled[-1], shuffled[0]
        self.pool = shuffled

    def next(self) -> Any:
        if not self.items:
            return None
        if not self.pool:
            self._refill()
        item = self.pool.pop(0)
        self.last_item = item
        return item


@dataclass
class CompositionResult:
    audio: np.ndarray                 # Mastered stereo float32 array, shape (2, N)
    sr: int                           # 44100
    duration: float                   # ~60s (55-65s)
    seed: int                         # Seed used
    scale: MusicalScale               # Scale used
    tempo_bpm: float                  # Tempo
    arrangement: CompositionArrangement
    score: CandidateScore
    stems_info: Dict[str, str]        # Active musical stems
    palette_info: Dict[str, int]      # Slice counts per acoustic role
    artist_name: str = ""
    artist_id: str = ""
    artist_track_hint: str = ""


def _tile_or_loop_bed(slices: List[AudioSlice], target_samples: int, sr: int, rng: SeededRNG) -> np.ndarray:
    """
    Creates a continuous atmospheric stereo bed traversing source slices
    chronologically across the timeline without clicks or sudden volume jumps.
    """
    if not slices:
        return np.zeros((2, target_samples), dtype=np.float32)

    bed_mono = np.zeros(target_samples, dtype=np.float32)
    cur_pos = 0
    slice_idx = 0

    # Ensure slices are traversed chronologically by start_sec
    sorted_slices = sorted(slices, key=lambda s: s.start_sec)

    while cur_pos < target_samples:
        sl = sorted_slices[slice_idx % len(sorted_slices)]
        slice_idx += 1
        s_audio = sl.audio.astype(np.float32)
        s_len = len(s_audio)
        if s_len <= 32:
            continue

        # Smooth crossfade length: proportional to slice length, up to 0.35s
        fade_len = max(128, min(int(0.35 * sr), int(s_len * 0.30)))
        t_fade = np.linspace(0.0, np.pi * 0.5, fade_len, dtype=np.float32)
        fade_in = np.sin(t_fade)
        fade_out = np.cos(t_fade)

        # Pre-window slice ends
        s_prepared = s_audio.copy()
        s_prepared[:fade_len] *= fade_in
        s_prepared[-fade_len:] *= fade_out

        end_pos = min(target_samples, cur_pos + s_len)
        write_len = end_pos - cur_pos
        if write_len <= 0:
            break

        bed_mono[cur_pos:end_pos] += s_prepared[:write_len]
        step = max(int(0.12 * sr), s_len - fade_len)
        cur_pos += step

    # Soft normalize bed and smooth beginning/end to ensure 100% zero crackle
    pk = np.max(np.abs(bed_mono))
    if pk > 1e-4:
        bed_mono = (bed_mono / pk) * 0.82

    edge_fade = min(int(0.04 * sr), target_samples // 4)
    if edge_fade > 1:
        att = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge_fade, dtype=np.float32))
        bed_mono[:edge_fade] *= att
        bed_mono[-edge_fade:] *= att[::-1]

    pan_offset = rng.uniform(-0.10, 0.10)
    return apply_panning(bed_mono, pan=pan_offset)


def render_candidate_composition(
    prep: PreprocessedAudio,
    analysis: CompleteAnalysis,
    seed: int,
    beat_preference: str = "pop",
    energy_preference: str = "low",
    palette: Optional[SourcePalette] = None,
    shared_pad: Optional[np.ndarray] = None,
    on_progress: Optional[Any] = None,
    target_duration: float = 30.0
) -> CompositionResult:
    """
    Renders a complete musical composition (30s or 60s) with punchy beats,
    rich bass, lush chords, and a catchy melodic hook forged from the source
    recording. Each genre is sonically distinct by design.
    """
    rng = SeededRNG(seed)
    sr = prep.sr

    # 1. Multi-Scale Palette Extraction (Reused across candidates if provided)
    if palette is None:
        palette = build_source_palette(
            audio=prep.mono,
            sr=sr,
            onset_samples=analysis.rhythm.onset_samples,
            target_slice_count=48
        )

    # 2. Select Iconic Artist Archetype & Musical Scale
    artist_profile = get_random_artist_for_genre(beat_preference, rng=rng)
    scale = select_scale(
        brightness=analysis.spectral.brightness,
        noisiness=analysis.texture.noisiness,
        candidate_root_notes=analysis.pitch.candidate_notes,
        rng=rng,
        genre_preference=beat_preference
    )

    # 3. Select Musical Tempo (BPM tailored to Artist Profile)
    bpm = rng.uniform(artist_profile.bpm_min, artist_profile.bpm_max)
    if energy_preference == "low":
        bpm = max(55.0, bpm * 0.90)
    elif energy_preference == "high":
        bpm = min(174.0, bpm * 1.06)
    bpm = round(bpm, 1)

    # 4. Plan Adaptive Song Arrangement (target 30s or 60s based on user choice)
    arrangement = plan_arrangement(
        bpm=bpm,
        source_energy_envelope=analysis.amplitude.envelope_curve,
        sonic_character=analysis.classification,
        rng=rng,
        target_sec=target_duration
    )
    total_duration = arrangement.total_duration
    total_samples = int(total_duration * sr)

    # ── Artist Archetype Sonic Parameters & Mix Settings ───────────────────────
    pref            = (beat_preference or "pop").lower()
    _lead_style     = artist_profile.lead_style
    _bass_style     = artist_profile.bass_style
    _chord_q        = artist_profile.chord_q
    _chord_reverb   = artist_profile.chord_reverb
    _melody_reverb  = artist_profile.lead_reverb
    _melody_delay   = artist_profile.lead_delay
    _bed_gain_mul   = artist_profile.bed_gain_mul
    _melody_mix     = artist_profile.melody_mix
    _chord_mix      = artist_profile.chord_mix
    _bass_mix       = artist_profile.bass_mix

    # 5. Generate Chord Progression & Melodic Hook (Genre-adapted)
    chords = select_chord_progression(scale, rng=rng, genre_preference=beat_preference)
    melody_events = sequence_melody(
        scale=scale,
        total_bars=arrangement.total_bars,
        seconds_per_bar=arrangement.seconds_per_bar,
        source_peaks=analysis.spectral.dominant_frequencies,
        has_pitch=analysis.pitch.has_reliable_pitch,
        density_factor=0.65 if energy_preference == "high" else 0.50,
        rng=rng,
        genre_preference=beat_preference
    )

    # 6. Generate Punchy Drum & Percussion Rhythm
    rhythm_track = generate_rhythm(
        style_preference=beat_preference,
        bpm=bpm,
        has_source_rhythm=analysis.rhythm.has_reliable_rhythm,
        rng=rng,
        artist_profile=artist_profile
    )

    # 7. Low-Memory Direct Accumulation Mix Buffer (shape: 2, total_samples)
    mix = np.zeros((2, total_samples), dtype=np.float32)

    # --- STEM 1: ATMOSPHERIC SOURCE BED (Full Timeline Chronological Traversal) ---
    bed_source_slices = palette.chronological_slices if palette.chronological_slices else (palette.ambience or palette.drones)
    raw_bed = _tile_or_loop_bed(bed_source_slices, total_samples, sr, rng=rng)

    bed_gain_curve = np.zeros(total_samples, dtype=np.float32)
    ramp_len = int(0.035 * sr)
    for i, sec in enumerate(arrangement.sections):
        s_start = int(sec.start_time * sr)
        s_end = min(total_samples, int((sec.start_time + sec.duration) * sr))
        target_gain = (0.50 + 0.30 * sec.energy_level) * _bed_gain_mul
        if s_end > s_start:
            bed_gain_curve[s_start:s_end] = target_gain
            if i > 0 and s_start > 0:
                prev_gain = (0.50 + 0.30 * arrangement.sections[i - 1].energy_level) * _bed_gain_mul
                actual_ramp = min(ramp_len, s_end - s_start, s_start)
                if actual_ramp > 1:
                    t_ramp = np.linspace(0.0, np.pi, actual_ramp, dtype=np.float32)
                    w = (0.5 - 0.5 * np.cos(t_ramp)).astype(np.float32)
                    r_start = s_start - actual_ramp // 2
                    r_end = r_start + actual_ramp
                    if 0 <= r_start and r_end <= total_samples:
                        bed_gain_curve[r_start:r_end] = prev_gain * (1.0 - w) + target_gain * w
    mix += raw_bed * bed_gain_curve
    del raw_bed

    # --- STEM 2: HARMONIC CHORD PROGRESSION (Resonated across diverse timeline chunks) ---
    if on_progress:
        on_progress(66, f"Resonating {len(chords)}-chord harmonic progression across timeline ({scale.name})...")
    chord_buffers = []
    num_chords = len(chords)
    bar_samp = int(arrangement.seconds_per_bar * sr)
    chord_src_len = min(len(prep.mono), bar_samp)

    # Distribute chunk offsets across 0% to 100% of the input recording
    offsets = np.linspace(0, max(0, len(prep.mono) - chord_src_len), max(1, num_chords), dtype=int)
    for idx, ch in enumerate(chords):
        off = int(offsets[idx % len(offsets)])
        src_chunk = prep.mono[off:off + chord_src_len].copy()
        if len(src_chunk) < chord_src_len:
            reps = int(np.ceil(chord_src_len / max(1, len(src_chunk))))
            src_chunk = np.tile(src_chunk, reps)[:chord_src_len]

        # De-click and RMS-level the excitation source
        src_chunk = apply_slice_envelope(src_chunk, attack_ms=6.0, release_ms=14.0, sr=sr)
        src_chunk = normalize_slice_rms(src_chunk, target_rms=0.14)

        ch_base = resonant_filter_bank(
            src_chunk,
            frequencies=ch.frequencies,
            q=_chord_q,
            sr=sr,
            envelope_shaping=True
        )
        # Ensure consistent chord volume across distinct source chunks
        ch_base = normalize_slice_rms(ch_base, target_rms=0.18)
        chord_buffers.append(ch_base)

    track_chords = np.zeros((2, total_samples), dtype=np.float32)
    sec_bar_offset = 0
    fade_samples = min(int(0.06 * sr), bar_samp // 4)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    for sec in arrangement.sections:
        if "chords" in sec.active_layers:
            for b in range(sec.bars):
                global_bar = sec_bar_offset + b
                chord_idx = global_bar % len(chords)
                b_start = int((sec.start_time + b * arrangement.seconds_per_bar) * sr)
                b_end = min(total_samples, b_start + bar_samp)
                b_len = b_end - b_start
                raw_c = chord_buffers[chord_idx]
                if len(raw_c) < b_len:
                    fade_seam = min(int(0.02 * sr), len(raw_c) // 4)
                    if fade_seam > 1:
                        raw_c_win = raw_c.copy()
                        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_seam, dtype=np.float32))
                        raw_c_win[:fade_seam] *= w
                        raw_c_win[-fade_seam:] *= w[::-1]
                    else:
                        raw_c_win = raw_c
                    reps = int(np.ceil(b_len / max(1, len(raw_c_win))))
                    c_chunk = np.tile(raw_c_win, reps)[:b_len].copy()
                else:
                    c_chunk = raw_c[:b_len].copy()

                if fade_samples > 1 and b_len >= fade_samples * 2:
                    c_chunk[:fade_samples] *= fade_in
                    c_chunk[-fade_samples:] *= fade_out

                pan_c = rng.uniform(-0.25, 0.25)
                panned_c = apply_panning(c_chunk * (0.78 * sec.energy_level), pan=pan_c)
                track_chords[:, b_start:b_end] += panned_c
        sec_bar_offset += sec.bars
    del chord_buffers

    # Blend with granular pad cloud for depth
    if shared_pad is not None:
        if shared_pad.shape[1] >= total_samples:
            pad_stereo = shared_pad[:, :total_samples]
        else:
            reps = int(np.ceil(total_samples / max(1, shared_pad.shape[1])))
            pad_stereo = np.tile(shared_pad, (1, reps))[:, :total_samples]
    else:
        pad_stereo = create_granular_pad(
            prep.mono,
            target_duration=total_duration,
            semitone_shift=0.0,
            grain_duration=0.16,
            density=8.0,
            sr=sr,
            stereo_spread=True
        )

    for sec in arrangement.sections:
        if "chords" in sec.active_layers:
            s_start = int(sec.start_time * sr)
            s_end = min(total_samples, int((sec.start_time + sec.duration) * sr))
            if s_end > s_start:
                sec_pad = pad_stereo[:, s_start:s_end] * (0.42 * sec.energy_level)
                sec_pad = apply_fade(sec_pad, fade_samples=min(int(0.04 * sr), (s_end - s_start) // 4))
                track_chords[:, s_start:s_end] += sec_pad
    del pad_stereo

    _c_room, _c_wet = _chord_reverb
    track_chords = schroeder_reverb(track_chords, room_size=_c_room, wet_level=_c_wet, sr=sr)

    # Daft Punk French Touch resonant filter sweep across chords in Rap mode
    if pref in ["rap", "trap", "drill"]:
        t_arr = np.linspace(0, total_duration, total_samples, endpoint=False, dtype=np.float32)
        sweep_lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * t_arr / max(2.0, arrangement.seconds_per_bar * 2))
        b_dp, a_dp = signal.butter(1, min(0.45, 2600.0 / (sr * 0.5)), btype='low')
        track_chords[0] = signal.lfilter(b_dp, a_dp, track_chords[0])
        track_chords[1] = signal.lfilter(b_dp, a_dp, track_chords[1])
        track_chords *= (0.72 + 0.38 * sweep_lfo)

    mix += track_chords * _chord_mix
    del track_chords

    # --- STEM 3: PUNCHY DRUMS & PERCUSSION (Full-Timeline Shuffled Selection) ---
    if on_progress:
        on_progress(74, f"Forging punchy noise kick, snare, open hats & groove ({scale.name})...")
    kick_times: List[int] = []

    if beat_preference != "none":
        track_drums = np.zeros((2, total_samples), dtype=np.float32)

        # Build varied genre-tailored one-shot pools from impacts and acoustic movements
        kick_samples = [render_noise_kick(sl.audio, velocity=0.96, sr=sr, genre=pref) for sl in palette.impacts[:4]]
        snare_sources = (palette.impacts[:3] + palette.movements[:2]) or palette.all_slices[:3]
        snare_samples = [render_noise_snare(sl.audio, velocity=0.90, sr=sr, genre=pref) for sl in snare_sources]
        hihat_samples = [render_noise_hihat(sl.audio, velocity=0.74, sr=sr, genre=pref) for sl in (palette.pulses[:5] or palette.all_slices[:5])]
        open_hihat_samples = [render_noise_open_hihat(sl.audio, velocity=0.78, sr=sr, genre=pref) for sl in (palette.pulses[:4] or palette.all_slices[:4])]

        kick_bag = BagSelector(kick_samples, rng=rng)
        snare_bag = BagSelector(snare_samples, rng=rng)
        hihat_bag = BagSelector(hihat_samples, rng=rng)
        open_hihat_bag = BagSelector(open_hihat_samples, rng=rng)
        foley_bag = BagSelector(palette.impacts + palette.textures, rng=rng)

        sec_bar_offset = 0
        step_dur = arrangement.seconds_per_bar / 16.0

        for sec in arrangement.sections:
            if "drums" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    pat_bar = global_bar % 4
                    bar_start_s = int((sec.start_time + b * arrangement.seconds_per_bar) * sr)

                    for step in range(16):
                        step_idx = pat_bar * 16 + step
                        step_s = bar_start_s + int(step * step_dur * sr)
                        if getattr(rhythm_track, 'humanize_timing_ms', 0.0) > 0.0:
                            j_ms = rhythm_track.humanize_timing_ms
                            j_samp = int(rng.uniform(-j_ms, j_ms) * 0.001 * sr)
                            step_s = max(0, min(total_samples - 1, step_s + j_samp))
                        if step_s >= total_samples:
                            continue

                        # 1. Kick Drum
                        k_vel = rhythm_track.kick_pattern[step_idx]
                        if k_vel > 0:
                            base_kick = kick_bag.next()
                            if base_kick is not None:
                                kl = min(len(base_kick), total_samples - step_s)
                                kick_hit = apply_fade(base_kick[:kl], fade_samples=32) * (k_vel * sec.energy_level)
                                track_drums[0, step_s:step_s + kl] += kick_hit
                                track_drums[1, step_s:step_s + kl] += kick_hit
                                kick_times.append(step_s)

                        # 2. Snare / Clap
                        s_vel = rhythm_track.snare_pattern[step_idx]
                        if s_vel > 0:
                            base_snare = snare_bag.next()
                            if base_snare is not None:
                                sl = min(len(base_snare), total_samples - step_s)
                                snare_hit = apply_fade(base_snare[:sl], fade_samples=32) * (s_vel * sec.energy_level)
                                track_drums[0, step_s:step_s + sl] += snare_hit
                                track_drums[1, step_s:step_s + sl] += snare_hit

                        # 3. Closed Hi-Hat
                        h_vel = rhythm_track.hihat_pattern[step_idx]
                        if h_vel > 0:
                            base_hihat = hihat_bag.next()
                            if base_hihat is not None:
                                hl = min(len(base_hihat), total_samples - step_s)
                                hihat_hit = apply_fade(base_hihat[:hl], fade_samples=24) * (h_vel * sec.energy_level)
                                pan_h = rng.uniform(-0.25, 0.25)
                                panned_h = apply_panning(hihat_hit, pan=pan_h)
                                track_drums[:, step_s:step_s + hl] += panned_h

                        # 4. Open Hi-Hat
                        if getattr(rhythm_track, 'open_hihat_pattern', None) is not None:
                            oh_vel = rhythm_track.open_hihat_pattern[step_idx]
                            if oh_vel > 0:
                                base_open_hihat = open_hihat_bag.next()
                                if base_open_hihat is not None:
                                    ohl = min(len(base_open_hihat), total_samples - step_s)
                                    open_hit = apply_fade(base_open_hihat[:ohl], fade_samples=32) * (oh_vel * sec.energy_level)
                                    pan_oh = rng.uniform(0.15, 0.35)
                                    panned_oh = apply_panning(open_hit, pan=pan_oh)
                                    track_drums[:, step_s:step_s + ohl] += panned_oh

                        # 5. Source Foley Percussion Chops
                        sp_vel = rhythm_track.source_perc_pattern[step_idx]
                        if sp_vel > 0:
                            foley_sl = foley_bag.next()
                            if foley_sl is not None:
                                fl = min(len(foley_sl.audio), total_samples - step_s, int(0.18 * sr))
                                if fl > 0:
                                    f_audio = apply_slice_envelope(foley_sl.audio[:fl], attack_ms=2.0, release_ms=6.0, sr=sr)
                                    f_audio = apply_fade(f_audio, fade_samples=24)
                                    pan_f = rng.uniform(-0.45, 0.45)
                                    panned_f = apply_panning(f_audio * sp_vel * 0.70, pan=pan_f)
                                    track_drums[:, step_s:step_s + fl] += panned_f

            sec_bar_offset += sec.bars

        # Sidechain Ducking: Smooth continuous duck curve on each kick hit (zero crackle/click)
        duck_len = int(0.12 * sr)
        attack_len = max(16, int(0.008 * sr))
        release_len = max(32, duck_len - attack_len)
        t_att = np.linspace(0, np.pi, attack_len, endpoint=False)
        duck_attack = 1.0 - 0.30 * 0.5 * (1.0 - np.cos(t_att))
        t_rel = np.linspace(0, 1.0, release_len)
        duck_release = 0.70 + 0.30 * (1.0 - np.exp(-4.0 * t_rel)) / (1.0 - np.exp(-4.0))
        duck_curve = np.concatenate([duck_attack, duck_release]).astype(np.float32)

        for ks in kick_times:
            d_end = min(total_samples, ks + duck_len)
            dl = d_end - ks
            if dl > 0:
                mix[:, ks:d_end] *= duck_curve[:dl]

        mix += track_drums * 0.95
        del track_drums

    # --- STEM 4: DEEP GROOVING SUB-BASS (Fused with Timeline Low-End Movements) ---
    if on_progress:
        on_progress(82, "Carving deep analog-saturated 808 sub-bass...")
    track_bass = np.zeros((2, total_samples), dtype=np.float32)
    b_lp, a_lp = signal.butter(2, min(0.45, 320.0 / (sr * 0.5)), btype='low')
    prefiltered_bass_source = signal.lfilter(b_lp, a_lp, prep.mono)
    p_src = np.max(np.abs(prefiltered_bass_source))
    if p_src > 1e-4:
        prefiltered_bass_source = prefiltered_bass_source / p_src

    bass_slices = (palette.drones + palette.tonal + palette.impacts) or palette.all_slices
    bass_slice_bag = BagSelector(bass_slices, rng=rng)
    sec_bar_offset = 0
    if _bass_style == "808_glide":
        # Saturated Pitch-Gliding 808 Sub-Bass (Metro Boomin, Usher Crunk, Atlanta Trap, Reggaeton)
        step_8_sec = arrangement.seconds_per_bar / 8.0
        note_dur = step_8_sec * 1.85

        for sec in arrangement.sections:
            if "bass" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    chord_idx = global_bar % len(chords)
                    chord = chords[chord_idx]
                    bar_time = sec.start_time + b * arrangement.seconds_per_bar
                    root_f = midi_to_hz(chord.root_midi)
                    oct_f = midi_to_hz(chord.root_midi + 12)
                    fifth_f = midi_to_hz(chord.root_midi + 7)

                    trap_steps = [
                        (0, root_f * (1.5 if getattr(artist_profile, "id", "") == "usher_liljon_crunk" else 1.0), root_f, 0.98),
                        (3, root_f, fifth_f, 0.88),
                        (6, root_f, oct_f, 0.95)
                    ]
                    for s8, f_st, f_nd, vel_b in trap_steps:
                        s_start = int((bar_time + s8 * step_8_sec) * sr)
                        if s_start >= total_samples:
                            continue

                        b_sl = bass_slice_bag.next()
                        b_audio = b_sl.audio if b_sl is not None else None
                        rendered_bass = render_trap_808_glide_bass(
                            source_audio=prefiltered_bass_source,
                            freq_start=f_st,
                            freq_end=f_nd,
                            duration=note_dur,
                            velocity=vel_b * sec.energy_level,
                            sr=sr,
                            glide_sec=0.045 if getattr(artist_profile, "id", "") == "usher_liljon_crunk" else 0.075,
                            slice_audio=b_audio
                        )
                        bl = min(len(rendered_bass), total_samples - s_start)
                        if bl > 0:
                            b_hit = apply_fade(rendered_bass[:bl], fade_samples=48)
                            track_bass[0, s_start:s_start + bl] += b_hit
                            track_bass[1, s_start:s_start + bl] += b_hit
            sec_bar_offset += sec.bars

    elif _bass_style in ["karplus_bass", "karplus_slap"]:
        # Karplus-Strong physical modeling bass guitar / funk slap bass (Michael Jackson, Bruno Mars, Dua Lipa, J Dilla)
        step_16_sec = arrangement.seconds_per_bar / 16.0
        note_dur = step_16_sec * 1.6
        is_slap = (_bass_style == "karplus_slap")
        damp = 0.992 if is_slap else 0.996

        for sec in arrangement.sections:
            if "bass" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    chord_idx = global_bar % len(chords)
                    chord = chords[chord_idx]
                    bar_time = sec.start_time + b * arrangement.seconds_per_bar
                    root_f = midi_to_hz(chord.root_midi)
                    while root_f > 115.0:
                        root_f /= 2.0
                    while root_f < 38.0:
                        root_f *= 2.0
                    oct_f = root_f * 2.0
                    fifth_f = root_f * 1.5

                    funk_hits = [
                        (0, root_f, 0.96),
                        (3, root_f, 0.86),
                        (6, fifth_f, 0.90),
                        (8, root_f, 0.92),
                        (10, oct_f, 0.94),
                        (14, fifth_f, 0.88)
                    ]
                    for s16, f_n, vel_b in funk_hits:
                        s_start = int((bar_time + s16 * step_16_sec) * sr)
                        if s_start >= total_samples:
                            continue
                        b_sl = bass_slice_bag.next()
                        b_grain = b_sl.audio if (b_sl is not None and len(b_sl.audio) > 0) else prefiltered_bass_source[:int(0.08 * sr)]
                        rendered_bass = render_karplus_strong_noise_note(
                            source_grain=b_grain,
                            freq=f_n,
                            duration=note_dur,
                            velocity=vel_b * sec.energy_level,
                            damping=damp,
                            brightness=0.62 if is_slap else 0.50,
                            sr=sr,
                            is_slap=is_slap
                        )
                        bl = min(len(rendered_bass), total_samples - s_start)
                        if bl > 0:
                            b_hit = apply_fade(rendered_bass[:bl], fade_samples=48)
                            track_bass[0, s_start:s_start + bl] += b_hit
                            track_bass[1, s_start:s_start + bl] += b_hit
            sec_bar_offset += sec.bars

    elif _bass_style == "saw_pluck":
        # Lady Gaga / RedOne: Sawtooth sub pluck synced to kick, short decay, no glide
        step_16_sec = arrangement.seconds_per_bar / 16.0
        note_dur = step_16_sec * 0.85

        for sec in arrangement.sections:
            if "bass" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    chord_idx = global_bar % len(chords)
                    chord = chords[chord_idx]
                    bar_time = sec.start_time + b * arrangement.seconds_per_bar
                    root_f = midi_to_hz(chord.root_midi)
                    oct_f = midi_to_hz(chord.root_midi + 12)

                    for s16 in range(16):
                        if s16 in [0, 4, 8, 12]:
                            f_note = root_f
                            vel_b = 0.98 * sec.energy_level
                        elif s16 in [2, 6, 10, 14]:
                            f_note = oct_f
                            vel_b = 0.86 * sec.energy_level
                        else:
                            continue

                        s_start = int((bar_time + s16 * step_16_sec) * sr)
                        if s_start >= total_samples:
                            continue

                        b_sl = bass_slice_bag.next()
                        b_audio = b_sl.audio if b_sl is not None else None
                        rendered_bass = render_noise_bass_note(
                            source_audio=prefiltered_bass_source,
                            freq=f_note,
                            duration=note_dur,
                            velocity=vel_b,
                            sr=sr,
                            is_prefiltered=True,
                            slice_audio=b_audio
                        )
                        bl = min(len(rendered_bass), total_samples - s_start)
                        if bl > 0:
                            b_hit = apply_fade(rendered_bass[:bl], fade_samples=48)
                            track_bass[0, s_start:s_start + bl] += b_hit
                            track_bass[1, s_start:s_start + bl] += b_hit
            sec_bar_offset += sec.bars

    elif _bass_style == "sub_octave_pulse":
        # The Weeknd / Max Martin: Sub-octave pulse doubling root motion, driving 8ths, tight and punchy
        step_8_sec = arrangement.seconds_per_bar / 8.0
        note_dur = step_8_sec * 0.88

        for sec in arrangement.sections:
            if "bass" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    chord_idx = global_bar % len(chords)
                    chord = chords[chord_idx]
                    bar_time = sec.start_time + b * arrangement.seconds_per_bar
                    root_f = midi_to_hz(chord.root_midi)

                    for s8 in range(8):
                        s_start = int((bar_time + s8 * step_8_sec) * sr)
                        if s_start >= total_samples:
                            continue
                        b_sl = bass_slice_bag.next()
                        b_audio = b_sl.audio if b_sl is not None else None
                        rendered_bass = render_noise_bass_note(
                            source_audio=prefiltered_bass_source,
                            freq=root_f,
                            duration=note_dur,
                            velocity=0.94 * sec.energy_level,
                            sr=sr,
                            is_prefiltered=True,
                            slice_audio=b_audio
                        )
                        bl = min(len(rendered_bass), total_samples - s_start)
                        if bl > 0:
                            b_hit = apply_fade(rendered_bass[:bl], fade_samples=48)
                            track_bass[0, s_start:s_start + bl] += b_hit
                            track_bass[1, s_start:s_start + bl] += b_hit
            sec_bar_offset += sec.bars

    elif _bass_style == "horn_punctuation":
        # Dr. Dre / Scott Storch: Sparse horn-synth bass punctuation, short percussive envelope
        step_8_sec = arrangement.seconds_per_bar / 8.0
        note_dur = 0.38

        for sec in arrangement.sections:
            if "bass" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    chord_idx = global_bar % len(chords)
                    chord = chords[chord_idx]
                    bar_time = sec.start_time + b * arrangement.seconds_per_bar
                    root_f = midi_to_hz(chord.root_midi)

                    for s8, vel_b in [(0, 0.98), (6, 0.88)]:
                        s_start = int((bar_time + s8 * step_8_sec) * sr)
                        if s_start >= total_samples:
                            continue
                        b_sl = bass_slice_bag.next()
                        b_audio = b_sl.audio if b_sl is not None else None
                        rendered_bass = render_noise_bass_note(
                            source_audio=prefiltered_bass_source,
                            freq=root_f,
                            duration=note_dur,
                            velocity=vel_b * sec.energy_level,
                            sr=sr,
                            is_prefiltered=True,
                            slice_audio=b_audio
                        )
                        bl = min(len(rendered_bass), total_samples - s_start)
                        if bl > 0:
                            b_hit = apply_fade(rendered_bass[:bl], fade_samples=48)
                            track_bass[0, s_start:s_start + bl] += b_hit
                            track_bass[1, s_start:s_start + bl] += b_hit
            sec_bar_offset += sec.bars

    else:
        # Standard warm sub-bass or ambient sustained sub
        step_8_sec = arrangement.seconds_per_bar / 8.0
        for sec in arrangement.sections:
            if "bass" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    chord_idx = global_bar % len(chords)
                    chord = chords[chord_idx]
                    bar_time = sec.start_time + b * arrangement.seconds_per_bar
                    root_f = midi_to_hz(chord.root_midi)
                    fifth_f = midi_to_hz(chord.root_midi + 7)

                    hits = [(0.0, root_f, 0.94)]
                    if _bass_style != "sub_drone":
                        hits.append((arrangement.seconds_per_bar * 0.5, fifth_f, 0.88))

                    b_dur = arrangement.seconds_per_bar * (0.92 if _bass_style == "sub_drone" else 0.45)
                    for b_sec, f_note, vel_b in hits:
                        b_start = int((bar_time + b_sec) * sr)
                        if b_start >= total_samples:
                            continue
                        b_sl = bass_slice_bag.next()
                        b_audio = b_sl.audio if b_sl is not None else None
                        rendered_bass = render_noise_bass_note(
                            source_audio=prefiltered_bass_source,
                            freq=f_note,
                            duration=b_dur,
                            velocity=vel_b * sec.energy_level,
                            sr=sr,
                            is_prefiltered=True,
                            slice_audio=b_audio
                        )
                        bl = min(len(rendered_bass), total_samples - b_start)
                        if bl > 0:
                            b_hit = apply_fade(rendered_bass[:bl], fade_samples=48)
                            track_bass[0, b_start:b_start + bl] += b_hit
                            track_bass[1, b_start:b_start + bl] += b_hit
            sec_bar_offset += sec.bars

    del prefiltered_bass_source

    # Clean Sidechain Ducking on Bass:
    # Smoothly dip sub-bass on every kick impact so kick and sub-bass never clash or clip
    if len(kick_times) > 0:
        duck_len_b = int(0.12 * sr)
        attack_len_b = max(16, int(0.008 * sr))
        release_len_b = max(32, duck_len_b - attack_len_b)
        t_att_b = np.linspace(0, np.pi, attack_len_b, endpoint=False)
        duck_att_b = 1.0 - 0.40 * 0.5 * (1.0 - np.cos(t_att_b))
        t_rel_b = np.linspace(0, 1.0, release_len_b)
        duck_rel_b = 0.60 + 0.40 * (1.0 - np.exp(-4.5 * t_rel_b)) / (1.0 - np.exp(-4.5))
        duck_curve_bass = np.concatenate([duck_att_b, duck_rel_b]).astype(np.float32)

        for ks in kick_times:
            d_end = min(total_samples, ks + duck_len_b)
            dl = d_end - ks
            if dl > 0:
                track_bass[:, ks:d_end] *= duck_curve_bass[:dl]

    bass_drive = 1.15
    if pref in ["trap", "drill"]:
        bass_drive = 1.50
    elif pref in ["rap", "electro"]:
        bass_drive = 1.35
    elif pref in ["hiphop", "hip_hop", "lofi"]:
        bass_drive = 1.22
    elif pref in ["pop", "dance"]:
        bass_drive = 1.20
    elif pref in ["pop", "dance"]:
        bass_drive = 1.22
    track_bass = soft_saturation(track_bass, drive=bass_drive)
    mix += track_bass * _bass_mix
    del track_bass

    # --- STEM 5: CATCHY MELODIC LEAD / HOOK (Rotated Grains across Full Timeline) ---
    track_melody = np.zeros((2, total_samples), dtype=np.float32)
    lead_style = _lead_style
    if on_progress:
        on_progress(88, f"Synthesizing tuned melodic hook ({lead_style.title()}) & ping-pong delay...")

    tonal_grains = [s.audio for s in palette.tonal] if palette.tonal else [s.audio for s in (palette.pulses + palette.impacts)]
    melody_grain_bag = BagSelector(tonal_grains, rng=rng)

    for event in melody_events:
        sec = next((s for s in arrangement.sections if s.start_time <= event.start_time < s.start_time + s.duration), None)
        if sec and "melody" in sec.active_layers:
            start_s = int(event.start_time * sr)
            if start_s < total_samples:
                cur_grain = melody_grain_bag.next()
                if cur_grain is None or len(cur_grain) == 0:
                    cur_grain = prep.mono[:min(len(prep.mono), int(0.08 * sr))]

                if lead_style == "karplus":
                    note_audio = render_karplus_strong_noise_note(
                        source_grain=cur_grain,
                        freq=event.freq_hz,
                        duration=event.duration,
                        velocity=event.velocity * sec.energy_level * 0.92,
                        damping=0.986,
                        brightness=0.60,
                        sr=sr
                    )
                else:
                    note_audio = render_noise_instrument_note(
                        source_audio=cur_grain,
                        freq=event.freq_hz,
                        duration=event.duration,
                        velocity=event.velocity * sec.energy_level * 0.88,
                        style=lead_style,
                        sr=sr
                    )
                pan = rng.uniform(-0.35, 0.35)
                panned_note = apply_panning(note_audio, pan=pan)
                nl = min(panned_note.shape[1], total_samples - start_s)
                if nl > 0:
                    track_melody[:, start_s:start_s + nl] += apply_fade(panned_note[:, :nl], fade_samples=48)

    _div, _fb, _dmix = _melody_delay
    _m_room, _m_wet = _melody_reverb
    track_melody = ping_pong_delay(track_melody, bpm=bpm, division=_div, feedback=_fb, mix=_dmix, sr=sr)
    track_melody = schroeder_reverb(track_melody, room_size=_m_room, wet_level=_m_wet, sr=sr)
    mix += track_melody * _melody_mix
    del track_melody


    # --- STEM 6: STRUCTURAL RISERS, DOWNLIFTERS & ACCENTS (Timeline Diversity) ---
    accent_bag = BagSelector(palette.accents if palette.accents else palette.impacts, rng=rng)

    for sec in arrangement.sections:
        # 1. Rising noise sweep leading up to Chorus_Climax
        if "riser" in sec.active_layers:
            riser_dur = min(sec.duration, 3.5)
            r_off = int(len(prep.mono) * 0.5)
            riser_audio = render_noise_riser(prep.mono, duration=riser_dur, sr=sr, offset_sample=r_off)
            r_start = int((sec.start_time + sec.duration - riser_dur) * sr)
            rl = min(len(riser_audio), total_samples - r_start)
            if rl > 0 and r_start >= 0:
                panned_riser = apply_panning(apply_fade(riser_audio[:rl], fade_samples=48) * 0.85, pan=0.0)
                mix[:, r_start:r_start + rl] += panned_riser * 0.65

        # 2. Downlifter / Impact crash at Chorus_Climax drop
        if sec.name == "Chorus_Climax" and palette.impacts:
            down_sl = accent_bag.next()
            dl_audio = down_sl.audio if down_sl is not None else palette.impacts[0].audio
            downlifter = render_noise_downlifter(dl_audio, duration=2.0, sr=sr)
            d_start = int(sec.start_time * sr)
            dl = min(len(downlifter), total_samples - d_start)
            if dl > 0:
                panned_dl = apply_panning(apply_fade(downlifter[:dl], fade_samples=48) * 0.90, pan=0.0)
                mix[:, d_start:d_start + dl] += panned_dl * 0.60

        # 3. Dynamic foley accents on structural markers
        if "accents" in sec.active_layers:
            acc_slice = accent_bag.next()
            if acc_slice is not None:
                acc_s = int(sec.start_time * sr)
                al = min(len(acc_slice.audio), total_samples - acc_s)
                if al > 0:
                    acc_audio = apply_slice_envelope(acc_slice.audio[:al], attack_ms=4.0, release_ms=10.0, sr=sr)
                    panned_acc = apply_panning(apply_fade(acc_audio, fade_samples=48) * 0.85, pan=rng.uniform(-0.3, 0.3))
                    mix[:, acc_s:acc_s + al] += panned_acc * 0.50

    # 10. Clean Mastering Chain (LUFS -14.0, True Peak Limiter -0.5 dB)
    if on_progress:
        on_progress(94, "Mastering audio (LUFS -14 true peak limiter) & scoring...")

    # Safety soft saturation limiter to guarantee zero summing overflow or distortion before mastering
    pk_mix = float(np.max(np.abs(mix)))
    if pk_mix > 1.25:
        mix = (np.tanh(mix * 0.80) / np.tanh(0.80 * 1.25) * 1.25).astype(np.float32)

    mastered = master_audio(mix, target_lufs=-14.0, target_peak_db=-0.5, sr=sr)

    # 11. Objective Quality Scoring
    score = score_composition(
        output_audio=mastered,
        source_audio=prep.mono,
        source_usage_ratio=1.00,
        synthetic_audio_ratio=0.00,
        sr=sr
    )

    # Contextual stem descriptions reflecting instrument style & production sound
    lead_name = _lead_style.replace('_', ' ').title()
    bass_name = _bass_style.replace('_', ' ').title()
    groove_name = artist_profile.groove_type.replace('_', ' ').title()

    drum_desc = f"Procedural Drum Groove ({groove_name})"
    bass_desc = f"Deep Synthesized Bassline ({bass_name})"
    lead_desc = f"Procedural Melodic Hook ({lead_name})"

    stems_info = {
        "Drums": drum_desc,
        "Bass": bass_desc,
        "Chords": f"Dynamic 4-Chord Resonant Progression ({scale.name})",
        "Melody": lead_desc,
        "Bed": "Atmospheric Source Recording Bed",
        "Accents": "Dynamic Noise Riser & Impact Sweeps"
    }

    palette_info = {
        "impacts": len(palette.impacts),
        "pulses": len(palette.pulses),
        "movements": len(palette.movements),
        "textures": len(palette.textures),
        "drones": len(palette.drones),
        "ambience": len(palette.ambience),
        "accents": len(palette.accents),
        "total_slices": palette.total_slices_extracted,
        "chronological_slices": len(palette.chronological_slices)
    }

    return CompositionResult(
        audio=mastered,
        sr=sr,
        duration=round(total_duration, 2),
        seed=seed,
        scale=scale,
        tempo_bpm=bpm,
        arrangement=arrangement,
        score=score,
        stems_info=stems_info,
        palette_info=palette_info,
        artist_name=artist_profile.name,
        artist_id=artist_profile.id,
        artist_track_hint=artist_profile.track_title_hint
    )


def generate_candidates(
    prep: PreprocessedAudio,
    analysis: CompleteAnalysis,
    base_seed: Optional[int] = None,
    beat_preference: str = "pop",
    energy_preference: str = "low",
    num_candidates: int = 1,
    on_progress: Optional[Any] = None,
    target_duration: float = 30.0
) -> Tuple[CompositionResult, List[CompositionResult]]:
    """Generates candidates with independent seeds, ranked by score. Optimized for speed and low RAM."""
    if base_seed is None:
        base_seed = generate_seed()

    import gc
    import os

    # On Render/low-perf containers reduce upfront work while keeping full quality output
    _is_low_perf = bool(
        os.environ.get("RENDER")
        or os.environ.get("USE_TMP_STORAGE")
        or os.environ.get("LOW_PERF")
    )
    _slice_count = 24 if _is_low_perf else 36
    _pad_duration = min(target_duration, 24.0) if _is_low_perf else target_duration

    # Pre-extract palette once and reuse across all candidates
    if on_progress:
        on_progress(54, "Forging source palette & transient micro-chops...")
    palette = build_source_palette(
        audio=prep.mono,
        sr=prep.sr,
        onset_samples=analysis.rhythm.onset_samples,
        target_slice_count=_slice_count
    )

    # Pre-render shared pad once across candidates
    if on_progress:
        on_progress(60, "Synthesizing granular ambient cloud & pad...")
    shared_pad = create_granular_pad(
        prep.mono,
        target_duration=_pad_duration,
        semitone_shift=0.0,
        grain_duration=0.16,
        density=12.0,
        sr=prep.sr,
        stereo_spread=True
    )

    seeds = [base_seed]
    for _ in range(num_candidates - 1):
        seeds.append(generate_seed())

    candidates: List[CompositionResult] = []
    for idx, s in enumerate(seeds):
        if on_progress:
            pct = 64 + int((idx / max(1, num_candidates)) * 10)
            on_progress(pct, f"Procedurally synthesizing candidate {idx + 1} of {num_candidates}...")
        cand = render_candidate_composition(
            prep=prep,
            analysis=analysis,
            seed=s,
            beat_preference=beat_preference,
            energy_preference=energy_preference,
            palette=palette,
            shared_pad=shared_pad,
            on_progress=on_progress,
            target_duration=target_duration
        )
        candidates.append(cand)
        gc.collect()
        time.sleep(0.01)  # Yield CPU slice to OS scheduler to prevent UI stutter

    candidates.sort(key=lambda c: c.score.total_score, reverse=True)
    return candidates[0], candidates
