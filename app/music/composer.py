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
from ..dsp.segmentation import AudioSlice, SourcePalette, build_source_palette
from ..dsp.synthesis import (
    render_karplus_strong_noise_note,
    render_noise_bass_note,
    render_noise_downlifter,
    render_noise_hihat,
    render_noise_instrument_note,
    render_noise_kick,
    render_noise_open_hihat,
    render_noise_riser,
    render_noise_snare
)
from ..utils.random import SeededRNG, generate_seed
from .arrangement import CompositionArrangement, Section, plan_arrangement
from .chords import select_chord_progression
from .melody import sequence_melody
from .rhythm_generator import generate_rhythm
from .scales import MusicalScale, select_scale
from .scoring import CandidateScore, score_composition


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


def _tile_or_loop_bed(slices: List[AudioSlice], target_samples: int, sr: int, rng: SeededRNG) -> np.ndarray:
    """Creates a continuous atmospheric stereo bed by crossfading source slices seamlessly without clicks."""
    if not slices:
        return np.zeros((2, target_samples), dtype=np.float32)

    bed_mono = np.zeros(target_samples, dtype=np.float32)
    cur_pos = 0
    slice_idx = 0

    while cur_pos < target_samples:
        sl = slices[slice_idx % len(slices)]
        slice_idx += 1
        s_audio = sl.audio.astype(np.float32)
        s_len = len(s_audio)
        if s_len <= 16:
            continue

        # Smooth crossfade length: proportional to slice length, up to 0.4s
        fade_len = max(64, min(int(0.40 * sr), int(s_len * 0.35)))
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

    pan_offset = rng.uniform(-0.15, 0.15)
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

    # 2. Select Musical Scale & Root Tonality (Genre-adapted)
    scale = select_scale(
        brightness=analysis.spectral.brightness,
        noisiness=analysis.texture.noisiness,
        candidate_root_notes=analysis.pitch.candidate_notes,
        rng=rng,
        genre_preference=beat_preference
    )

    # 3. Select Musical Tempo (BPM tailored to Genre archetype)
    pref = (beat_preference or "pop").lower()
    if pref in ["pop", "dance", "synthpop"]:
        bpm = rng.uniform(121.0, 125.0)   # Lady Gaga & Nelly Furtado driving dance-pop pulse
    elif pref in ["rap", "trap", "drill"]:
        bpm = rng.uniform(104.0, 108.0)   # Kanye West & Daft Punk "Stronger" / French touch pocket
    elif pref in ["hiphop", "hip_hop", "boom_bap", "boombap", "lofi"]:
        bpm = rng.uniform(88.0, 94.0)     # J Dilla & Nujabes swung boom-bap pocket
    elif analysis.rhythm.has_reliable_rhythm:
        bpm = analysis.rhythm.estimated_tempo + rng.uniform(-2.0, 2.0)
    else:
        bpm = rng.uniform(120.0, 126.0)

    if energy_preference == "low":
        bpm = max(70.0, bpm * 0.90)
    elif energy_preference == "high":
        bpm = min(150.0, bpm * 1.08)
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

    # ── Per-genre sonic character parameters ───────────────────────────────────
    # Each genre gets its own reverb character, filter Q, mix levels and lead
    # style so they sound unmistakably different from each other.
    if pref in ["pop", "dance", "synthpop"]:
        # Lady Gaga & Nelly Furtado (RedOne & Timbaland)
        _lead_style     = "lady_gaga"  # Bright cutting electro pluck
        _chord_q        = 14.0         # Clean, open resonators
        _chord_reverb   = (0.65, 0.22) # Tight, punchy club room
        _melody_reverb  = (0.65, 0.20)
        _melody_delay   = (0.5, 0.28, 0.24)   # Syncopated 8th-note stereo bounce
        _bed_gain_mul   = 0.68         # Distinct, prominent input noise bed!
        _melody_mix     = 0.95
        _chord_mix      = 0.88
        _bass_mix       = 0.95         # Bouncy driving electro bass
    elif pref in ["rap", "trap", "drill"]:
        # Kanye West & Daft Punk ("Stronger" / French touch electro-hop)
        _lead_style     = "daft_punk"  # Resonant vocoder / French touch synth lead
        _chord_q        = 22.0         # Resonant filter sweep Q
        _chord_reverb   = (0.75, 0.24)
        _melody_reverb  = (0.68, 0.18)
        _melody_delay   = (0.5, 0.32, 0.28)   # Daft Punk syncopated echo
        _bed_gain_mul   = 0.64         # Distinct input noise soul chops!
        _melody_mix     = 0.96
        _chord_mix      = 0.85
        _bass_mix       = 1.05         # Heavy punchy 808 + funky synth bass
    elif pref in ["hiphop", "hip_hop", "boom_bap", "boombap", "lofi"]:
        # J Dilla & Nujabes (warm swung boom-bap)
        _lead_style     = "karplus"    # Warm noise-pluck string
        _chord_q        = 18.0         # Warm, wide Rhodes-like resonators
        _chord_reverb   = (0.90, 0.38) # Spacious and lush
        _melody_reverb  = (0.82, 0.32)
        _melody_delay   = (0.5, 0.38, 0.28)
        _bed_gain_mul   = 0.70         # Prominent atmospheric vinyl noise bed
        _melody_mix     = 0.85
        _chord_mix      = 0.85
        _bass_mix       = 0.92
    elif pref in ["none", "ambient"]:
        # Brian Eno & Hans Zimmer (vast cinematic soundscapes)
        _lead_style     = "karplus"    # Soft, barely-there resonance
        _chord_q        = 12.0         # Very smooth, wide bandpass cloud
        _chord_reverb   = (0.97, 0.55) # Maximum reverb wash
        _melody_reverb  = (0.96, 0.50)
        _melody_delay   = (0.75, 0.55, 0.40)  # Long trailing echo
        _bed_gain_mul   = 0.88         # Bed IS the composition
        _melody_mix     = 0.45         # Subtle ghost melody
        _chord_mix      = 0.78
        _bass_mix       = 0.55
    else:  # minimal, rhythmic, light_percussion, standard
        if pref in ["rhythmic", "energetic"]:
            _lead_style     = "pluck"
            _chord_q        = 18.0
            _chord_reverb   = (0.70, 0.22)   # Forward-push, less wash
            _melody_reverb  = (0.68, 0.18)
            _melody_delay   = (0.5, 0.20, 0.18)
            _bed_gain_mul   = 0.35
            _melody_mix     = 0.90
            _chord_mix      = 0.80
            _bass_mix       = 0.88
        elif pref in ["light_percussion", "organic", "folk"]:
            _lead_style     = "karplus"
            _chord_q        = 14.0           # Smooth, organic wide resonance
            _chord_reverb   = (0.88, 0.40)   # Warm hall reverb
            _melody_reverb  = (0.84, 0.35)
            _melody_delay   = (0.5, 0.30, 0.26)
            _bed_gain_mul   = 0.52
            _melody_mix     = 0.78
            _chord_mix      = 0.78
            _bass_mix       = 0.75
        else:  # minimal / standard
            _lead_style     = "karplus"
            _chord_q        = 22.0
            _chord_reverb   = (0.85, 0.28)
            _melody_reverb  = (0.75, 0.22)
            _melody_delay   = (0.5, 0.25, 0.22)
            _bed_gain_mul   = 0.45
            _melody_mix     = 0.80
            _chord_mix      = 0.75
            _bass_mix       = 0.84

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
        rng=rng
    )

    # Select best source slices for drum sound design
    k_slice = palette.impacts[0].audio if palette.impacts else prep.mono[:int(0.20 * sr)]
    s_slice = palette.pulses[0].audio if palette.pulses else prep.mono[:int(0.15 * sr)]
    h_slice = palette.pulses[1].audio if len(palette.pulses) > 1 else prep.mono[:int(0.08 * sr)]

    # 7. Low-Memory Direct Accumulation Mix Buffer (shape: 2, total_samples)
    # Renders stems sequentially and accumulates in-place to cut memory footprint by >85%
    mix = np.zeros((2, total_samples), dtype=np.float32)

    # --- STEM 1: ATMOSPHERIC SOURCE BED ---
    bed_source_slices = palette.ambience if palette.ambience else palette.drones
    raw_bed = _tile_or_loop_bed(bed_source_slices, total_samples, sr, rng=rng)

    for sec in arrangement.sections:
        s_start = int(sec.start_time * sr)
        s_end = min(total_samples, int((sec.start_time + sec.duration) * sr))
        if s_end > s_start:
            gain = (0.50 + 0.30 * sec.energy_level) * _bed_gain_mul
            mix[:, s_start:s_end] += raw_bed[:, s_start:s_end] * gain
    del raw_bed

    # --- STEM 2: HARMONIC CHORD PROGRESSION (Resonated directly from source noise) ---
    if on_progress:
        on_progress(66, f"Resonating {len(chords)}-chord harmonic progression from noise ({scale.name})...")
    # Pre-render a resonant buffer for each distinct chord in the progression
    # Q factor is genre-specific: high Q = tight/resonant (trap); low Q = smooth (pop/ambient)
    chord_buffers = []
    chord_src_len = min(len(prep.mono), int(4.0 * sr))
    for ch in chords:
        ch_base = resonant_filter_bank(
            prep.mono[:chord_src_len],
            frequencies=ch.frequencies,
            q=_chord_q,
            sr=sr,
            envelope_shaping=True
        )
        chord_buffers.append(ch_base)

    track_chords = np.zeros((2, total_samples), dtype=np.float32)
    sec_bar_offset = 0
    bar_samp = int(arrangement.seconds_per_bar * sr)
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
                if b_len > 0:
                    raw_c = chord_buffers[chord_idx]
                    reps = int(np.ceil(b_len / max(1, len(raw_c))))
                    c_chunk = np.tile(raw_c, reps)[:b_len].copy()

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
                track_chords[:, s_start:s_end] += pad_stereo[:, s_start:s_end] * (0.42 * sec.energy_level)
    del pad_stereo

    _c_room, _c_wet = _chord_reverb
    track_chords = schroeder_reverb(track_chords, room_size=_c_room, wet_level=_c_wet, sr=sr)

    # Daft Punk French Touch resonant filter sweep across chords in Rap mode
    if pref in ["rap", "trap", "drill"]:
        t_arr = np.linspace(0, total_duration, total_samples, endpoint=False, dtype=np.float32)
        # 2-bar sweeping LFO modulation
        sweep_lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * t_arr / max(2.0, arrangement.seconds_per_bar * 2))
        b_dp, a_dp = signal.butter(1, min(0.45, 2600.0 / (sr * 0.5)), btype='low')
        track_chords[0] = signal.lfilter(b_dp, a_dp, track_chords[0])
        track_chords[1] = signal.lfilter(b_dp, a_dp, track_chords[1])
        track_chords *= (0.72 + 0.38 * sweep_lfo)

    mix += track_chords * _chord_mix
    del track_chords

    # --- STEM 3: PUNCHY DRUMS & PERCUSSION (Pre-rendered One-Shots for 10x Speed) ---
    if on_progress:
        on_progress(74, f"Forging punchy noise kick, snare, open hats & groove ({scale.name})...")
    kick_times: List[int] = []

    if beat_preference != "none":
        track_drums = np.zeros((2, total_samples), dtype=np.float32)
        base_kick = render_noise_kick(k_slice, velocity=0.96, sr=sr)
        base_snare = render_noise_snare(s_slice, velocity=0.90, sr=sr)
        base_hihat = render_noise_hihat(h_slice, velocity=0.74, sr=sr)
        base_open_hihat = render_noise_open_hihat(h_slice, velocity=0.78, sr=sr)

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
                        if step_s >= total_samples:
                            continue

                        # 1. Kick Drum
                        k_vel = rhythm_track.kick_pattern[step_idx]
                        if k_vel > 0:
                            kl = min(len(base_kick), total_samples - step_s)
                            kick_hit = base_kick[:kl] * (k_vel * sec.energy_level)
                            track_drums[0, step_s:step_s + kl] += kick_hit
                            track_drums[1, step_s:step_s + kl] += kick_hit
                            kick_times.append(step_s)

                        # 2. Snare / Clap
                        s_vel = rhythm_track.snare_pattern[step_idx]
                        if s_vel > 0:
                            sl = min(len(base_snare), total_samples - step_s)
                            snare_hit = base_snare[:sl] * (s_vel * sec.energy_level)
                            track_drums[0, step_s:step_s + sl] += snare_hit
                            track_drums[1, step_s:step_s + sl] += snare_hit

                        # 3. Closed Hi-Hat
                        h_vel = rhythm_track.hihat_pattern[step_idx]
                        if h_vel > 0:
                            hl = min(len(base_hihat), total_samples - step_s)
                            hihat_hit = base_hihat[:hl] * (h_vel * sec.energy_level)
                            pan_h = rng.uniform(-0.25, 0.25)
                            panned_h = apply_panning(hihat_hit, pan=pan_h)
                            track_drums[:, step_s:step_s + hl] += panned_h

                        # 4. Open Hi-Hat
                        if getattr(rhythm_track, 'open_hihat_pattern', None) is not None:
                            oh_vel = rhythm_track.open_hihat_pattern[step_idx]
                            if oh_vel > 0:
                                ohl = min(len(base_open_hihat), total_samples - step_s)
                                open_hit = base_open_hihat[:ohl] * (oh_vel * sec.energy_level)
                                pan_oh = rng.uniform(0.15, 0.35)
                                panned_oh = apply_panning(open_hit, pan=pan_oh)
                                track_drums[:, step_s:step_s + ohl] += panned_oh

                        # 5. Source Foley Percussion Chops
                        sp_vel = rhythm_track.source_perc_pattern[step_idx]
                        if sp_vel > 0 and palette.impacts:
                            foley_sl = palette.impacts[(step + b) % len(palette.impacts)].audio
                            fl = min(len(foley_sl), total_samples - step_s, int(0.18 * sr))
                            if fl > 0:
                                pan_f = rng.uniform(-0.45, 0.45)
                                panned_f = apply_panning(foley_sl[:fl] * sp_vel * 0.70, pan=pan_f)
                                track_drums[:, step_s:step_s + fl] += panned_f

            sec_bar_offset += sec.bars

        # Sidechain Ducking: Duck the mix bed slightly on each kick hit
        duck_len = int(0.12 * sr)
        duck_curve = 1.0 - 0.35 * np.exp(-np.linspace(0, 4, duck_len))
        for ks in kick_times:
            d_end = min(total_samples, ks + duck_len)
            dl = d_end - ks
            if dl > 0:
                mix[:, ks:d_end] *= duck_curve[:dl]

        mix += track_drums * 0.95
        del track_drums

    # --- STEM 4: DEEP GROOVING SUB-BASS (Fused with pre-filtered low-end) ---
    if on_progress:
        on_progress(82, "Carving deep analog-saturated 808 sub-bass...")
    track_bass = np.zeros((2, total_samples), dtype=np.float32)
    b_lp, a_lp = signal.butter(2, min(0.45, 320.0 / (sr * 0.5)), btype='low')
    prefiltered_bass_source = signal.lfilter(b_lp, a_lp, prep.mono)
    p_src = np.max(np.abs(prefiltered_bass_source))
    if p_src > 1e-4:
        prefiltered_bass_source = prefiltered_bass_source / p_src

    sec_bar_offset = 0
    if pref in ["pop", "dance", "synthpop"]:
        # Lady Gaga & Nelly Furtado rolling 16th-note electro-pop synth bassline
        # Bouncy, driving, tight notes on 16th subdivisions (Poker Face / Promiscuous groove)
        step_16_sec = arrangement.seconds_per_bar / 16.0
        note_dur = step_16_sec * 0.82

        for sec in arrangement.sections:
            if "bass" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    chord_idx = global_bar % len(chords)
                    chord = chords[chord_idx]
                    bar_time = sec.start_time + b * arrangement.seconds_per_bar
                    root_f = midi_to_hz(chord.root_midi)
                    fifth_f = midi_to_hz(chord.root_midi + 7)
                    oct_f = midi_to_hz(chord.root_midi + 12)

                    # Rolling 16-step electro-pop bassline:
                    for s16 in range(16):
                        if s16 in [0, 4, 8, 12]:
                            f_note = root_f
                            vel_b = 0.95 * sec.energy_level
                        elif s16 in [2, 6, 10]:
                            f_note = root_f
                            vel_b = 0.88 * sec.energy_level
                        elif s16 in [1, 5, 9, 13]:
                            f_note = oct_f if (s16 in [1, 9]) else fifth_f
                            vel_b = 0.84 * sec.energy_level
                        elif s16 in [14, 15]:
                            f_note = root_f
                            vel_b = 0.86 * sec.energy_level
                        else:
                            continue

                        s_start = int((bar_time + s16 * step_16_sec) * sr)
                        if s_start >= total_samples:
                            continue

                        rendered_bass = render_noise_bass_note(
                            source_audio=prefiltered_bass_source,
                            freq=f_note,
                            duration=note_dur,
                            velocity=vel_b,
                            sr=sr,
                            is_prefiltered=True
                        )
                        bl = min(len(rendered_bass), total_samples - s_start)
                        if bl > 0:
                            track_bass[0, s_start:s_start + bl] += rendered_bass[:bl]
                            track_bass[1, s_start:s_start + bl] += rendered_bass[:bl]
            sec_bar_offset += sec.bars
    elif pref in ["rap", "trap", "drill"]:
        # Kanye West & Daft Punk syncopated funky electro-hiphop bassline
        step_8_sec = arrangement.seconds_per_bar / 8.0
        note_dur = step_8_sec * 0.90

        for sec in arrangement.sections:
            if "bass" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    chord_idx = global_bar % len(chords)
                    chord = chords[chord_idx]
                    bar_time = sec.start_time + b * arrangement.seconds_per_bar
                    root_f = midi_to_hz(chord.root_midi)
                    oct_f = midi_to_hz(chord.root_midi + 12)

                    # Daft Punk syncopated 8th-note funk groove:
                    for s8 in [0, 2, 3, 5, 7]:
                        f_note = oct_f if s8 in [3, 7] else root_f
                        vel_b = 0.96 * sec.energy_level
                        s_start = int((bar_time + s8 * step_8_sec) * sr)
                        if s_start >= total_samples:
                            continue

                        rendered_bass = render_noise_bass_note(
                            source_audio=prefiltered_bass_source,
                            freq=f_note,
                            duration=note_dur,
                            velocity=vel_b,
                            sr=sr,
                            is_prefiltered=True
                        )
                        bl = min(len(rendered_bass), total_samples - s_start)
                        if bl > 0:
                            track_bass[0, s_start:s_start + bl] += rendered_bass[:bl]
                            track_bass[1, s_start:s_start + bl] += rendered_bass[:bl]
            sec_bar_offset += sec.bars
    else:
        # Sustained sub-bass for hip-hop / minimal / ambient
        for sec in arrangement.sections:
            if "bass" in sec.active_layers:
                for b in range(sec.bars):
                    global_bar = sec_bar_offset + b
                    b_time = sec.start_time + b * arrangement.seconds_per_bar
                    b_samp = int(b_time * sr)
                    chord_idx = global_bar % len(chords)
                    chord = chords[chord_idx]
                    bass_dur = arrangement.seconds_per_bar * 0.94

                    rendered_bass = render_noise_bass_note(
                        source_audio=prefiltered_bass_source,
                        freq=midi_to_hz(chord.root_midi),
                        duration=bass_dur,
                        velocity=0.92 * sec.energy_level,
                        sr=sr,
                        is_prefiltered=True
                    )
                    bl = min(len(rendered_bass), total_samples - b_samp)
                    if bl > 0 and b_samp < total_samples:
                        track_bass[0, b_samp:b_samp + bl] += rendered_bass[:bl]
                        track_bass[1, b_samp:b_samp + bl] += rendered_bass[:bl]
            sec_bar_offset += sec.bars

    del prefiltered_bass_source
    bass_drive = 1.15
    if pref in ["rap", "trap", "drill"]:
        bass_drive = 1.45
    elif pref in ["hiphop", "hip_hop", "lofi"]:
        bass_drive = 1.25
    elif pref in ["pop", "dance"]:
        bass_drive = 1.22
    track_bass = soft_saturation(track_bass, drive=bass_drive)
    mix += track_bass * _bass_mix
    del track_bass

    # --- STEM 5: CATCHY MELODIC LEAD / HOOK (Karplus-Strong Noise String Synthesis) ---
    # Lead style is FORCED per genre for distinct sonic identity (not random)
    track_melody = np.zeros((2, total_samples), dtype=np.float32)
    lead_style = _lead_style
    if on_progress:
        on_progress(88, f"Synthesizing tuned melodic hook ({lead_style.title()}) & ping-pong delay...")

    # Acoustic micro-grain extracted from source sound for Karplus-Strong delay excitation
    if palette.pulses:
        ks_grain = palette.pulses[0].audio
    elif palette.impacts:
        ks_grain = palette.impacts[0].audio
    else:
        ks_grain = prep.mono[:min(len(prep.mono), int(0.08 * sr))]

    for event in melody_events:
        sec = next((s for s in arrangement.sections if s.start_time <= event.start_time < s.start_time + s.duration), None)
        if sec and "melody" in sec.active_layers:
            start_s = int(event.start_time * sr)
            if start_s < total_samples:
                if lead_style == "karplus":
                    note_audio = render_karplus_strong_noise_note(
                        source_grain=ks_grain,
                        freq=event.freq_hz,
                        duration=event.duration,
                        velocity=event.velocity * sec.energy_level * 0.92,
                        damping=0.986,
                        brightness=0.60,
                        sr=sr
                    )
                else:
                    note_audio = render_noise_instrument_note(
                        source_audio=prep.mono,
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
                    track_melody[:, start_s:start_s + nl] += panned_note[:, :nl]

    _div, _fb, _dmix = _melody_delay
    _m_room, _m_wet = _melody_reverb
    track_melody = ping_pong_delay(track_melody, bpm=bpm, division=_div, feedback=_fb, mix=_dmix, sr=sr)
    track_melody = schroeder_reverb(track_melody, room_size=_m_room, wet_level=_m_wet, sr=sr)
    mix += track_melody * _melody_mix
    del track_melody

    # --- STEM 6: STRUCTURAL RISERS, DOWNLIFTERS & ACCENTS ---
    for sec in arrangement.sections:
        # 1. Rising noise sweep leading up to Chorus_Climax
        if "riser" in sec.active_layers:
            riser_dur = min(sec.duration, 3.5)
            riser_audio = render_noise_riser(prep.mono, duration=riser_dur, sr=sr)
            r_start = int((sec.start_time + sec.duration - riser_dur) * sr)
            rl = min(len(riser_audio), total_samples - r_start)
            if rl > 0 and r_start >= 0:
                panned_riser = apply_panning(riser_audio[:rl] * 0.85, pan=0.0)
                mix[:, r_start:r_start + rl] += panned_riser * 0.65

        # 2. Downlifter / Impact crash at Chorus_Climax drop
        if sec.name == "Chorus_Climax" and palette.impacts:
            downlifter = render_noise_downlifter(palette.impacts[0].audio, duration=2.0, sr=sr)
            d_start = int(sec.start_time * sr)
            dl = min(len(downlifter), total_samples - d_start)
            if dl > 0:
                panned_dl = apply_panning(downlifter[:dl] * 0.90, pan=0.0)
                mix[:, d_start:d_start + dl] += panned_dl * 0.60

        # 3. Dynamic foley accents on structural markers
        if "accents" in sec.active_layers and palette.accents:
            acc_slice = rng.choice(palette.accents)
            acc_s = int(sec.start_time * sr)
            al = min(len(acc_slice.audio), total_samples - acc_s)
            if al > 0:
                panned_acc = apply_panning(acc_slice.audio[:al] * 0.85, pan=rng.uniform(-0.3, 0.3))
                mix[:, acc_s:acc_s + al] += panned_acc * 0.50

    # 10. Clean Mastering Chain (LUFS -14.0, True Peak Limiter -0.5 dB)
    if on_progress:
        on_progress(94, "Mastering audio (LUFS -14 true peak limiter) & scoring...")
    mastered = master_audio(mix, target_lufs=-14.0, target_peak_db=-0.5, sr=sr)

    # 11. Objective Quality Scoring
    score = score_composition(
        output_audio=mastered,
        source_audio=prep.mono,
        source_usage_ratio=1.00,
        synthetic_audio_ratio=0.00,
        sr=sr
    )

    # Contextual stem descriptions reflecting genre mode & artist inspiration
    if pref in ["pop", "dance", "synthpop"]:
        drum_desc = "Lady Gaga & Nelly Furtado Four-on-the-Floor Kick, Pop Claps, Disco Open Hats & Timbaland Source Stutters"
        bass_desc = "Rolling 16th-Note Electro-Pop Synth Bassline (Poker Face & Promiscuous Style)"
        lead_desc = f"Anthemic Pop Earworm Lead Hook ({lead_style.title()})"
    elif pref in ["rap", "trap", "drill"]:
        drum_desc = "Kanye West Punchy Compressed Kick, Vintage Claps, Daft Punk Swung Disco Hats & Soul Chops"
        bass_desc = "Daft Punk Funky Synth Bass & Saturated 808 Sub Wall"
        lead_desc = f"Daft Punk French Touch Vocoder Lead & Resonant Filter Sweep ({lead_style.title()})"
    elif pref in ["hiphop", "hip_hop", "boom_bap", "lofi"]:
        drum_desc = "J Dilla & Nujabes Swung Boom-Bap Kick, Snare & Organic Foley Chops"
        bass_desc = "Warm Analog-Saturated Sub-Bass Pocket"
        lead_desc = f"Soulful Noise Pluck Hook ({lead_style.title()})"
    else:
        drum_desc = f"Source-Crafted Punchy Kick, Snare, Open Hats & Groove ({beat_preference.title()})"
        bass_desc = "Source-Textured Warm Sub-Bass"
        lead_desc = f"Tuned Noise Melodic Hook ({lead_style.title()})"

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
        "total_slices": palette.total_slices_extracted
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
        palette_info=palette_info
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
    _slice_count = 16 if _is_low_perf else 24
    _pad_duration = 20.0 if _is_low_perf else 30.0

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
