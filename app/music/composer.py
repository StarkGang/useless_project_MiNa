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
    render_noise_bass_note,
    render_noise_hihat,
    render_noise_instrument_note,
    render_noise_kick,
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
    """Creates a continuous atmospheric stereo bed by crossfading long source slices."""
    bed_mono = np.zeros(target_samples, dtype=np.float32)
    cur_pos = 0
    fade_len = int(0.5 * sr)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    slice_idx = 0
    while cur_pos < target_samples:
        sl = slices[slice_idx % len(slices)]
        slice_idx += 1
        s_audio = sl.audio
        s_len = len(s_audio)

        end_pos = min(target_samples, cur_pos + s_len)
        write_len = end_pos - cur_pos

        if write_len <= 0:
            break

        chunk = s_audio[:write_len].copy()
        if cur_pos > 0 and write_len >= fade_len:
            chunk[:fade_len] *= fade_in
            bed_mono[cur_pos:cur_pos + fade_len] *= fade_out

        bed_mono[cur_pos:end_pos] += chunk
        cur_pos += max(int(0.5 * sr), s_len - fade_len)

    pan_offset = rng.uniform(-0.15, 0.15)
    return apply_panning(bed_mono, pan=pan_offset)


def render_candidate_composition(
    prep: PreprocessedAudio,
    analysis: CompleteAnalysis,
    seed: int,
    beat_preference: str = "minimal",
    energy_preference: str = "balanced",
    palette: Optional[SourcePalette] = None,
    shared_pad: Optional[np.ndarray] = None
) -> CompositionResult:
    """
    Renders a complete ~60s musical composition with punchy beats, rich bass,
    lush chords, and a catchy melodic hook forged from the source recording.
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
    pref = (beat_preference or "minimal").lower()
    if pref in ["hiphop", "hip_hop", "boom_bap", "boombap", "lofi"]:
        bpm = rng.uniform(86.0, 96.0)     # Classic Hip Hop pocket
    elif pref in ["rap", "trap", "drill"]:
        bpm = rng.uniform(134.0, 146.0)   # Modern Trap half-time pocket
    elif pref in ["pop", "dance", "synthpop"]:
        bpm = rng.uniform(118.0, 128.0)   # Driving Pop radio pulse
    elif analysis.rhythm.has_reliable_rhythm:
        bpm = analysis.rhythm.estimated_tempo + rng.uniform(-2.0, 2.0)
    else:
        bpm = rng.uniform(84.0, 114.0)

    if energy_preference == "low":
        bpm = max(70.0, bpm * 0.90)
    elif energy_preference == "high":
        bpm = min(150.0, bpm * 1.08)
    bpm = round(bpm, 1)

    # 4. Plan Adaptive Song Arrangement strictly bounded to ~60s
    arrangement = plan_arrangement(
        bpm=bpm,
        source_energy_envelope=analysis.amplitude.envelope_curve,
        sonic_character=analysis.classification,
        rng=rng
    )
    total_duration = arrangement.total_duration
    total_samples = int(total_duration * sr)

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

    # 7. Initialize Multi-Track Musical Stems
    track_bed = np.zeros((2, total_samples), dtype=np.float32)
    track_drums = np.zeros((2, total_samples), dtype=np.float32)
    track_bass = np.zeros((2, total_samples), dtype=np.float32)
    track_chords = np.zeros((2, total_samples), dtype=np.float32)
    track_melody = np.zeros((2, total_samples), dtype=np.float32)
    track_accents = np.zeros((2, total_samples), dtype=np.float32)

    # --- STEM 1: ATMOSPHERIC SOURCE BED ---
    bed_source_slices = palette.ambience if palette.ambience else palette.drones
    raw_bed = _tile_or_loop_bed(bed_source_slices, total_samples, sr, rng=rng)

    for sec in arrangement.sections:
        s_start = int(sec.start_time * sr)
        s_end = min(total_samples, int((sec.start_time + sec.duration) * sr))
        if s_end > s_start:
            gain = 0.50 + 0.30 * sec.energy_level
            track_bed[:, s_start:s_end] = raw_bed[:, s_start:s_end] * gain

    # --- STEM 2: HARMONIC CHORDS & PADS (Resonated directly from source noise) ---
    tile_reps = int(np.ceil(total_samples / max(1, len(prep.mono))))
    tiled_source = np.tile(prep.mono, tile_reps)[:total_samples]

    chord_freqs = [midi_to_hz(m) for m in scale.get_notes_in_octave_range(48, 72)[:4]]
    resonated_mono = resonant_filter_bank(
        tiled_source,
        frequencies=chord_freqs,
        q=20.0,
        sr=sr,
        envelope_shaping=True
    )
    resonated_stereo = apply_panning(resonated_mono, pan=rng.uniform(-0.25, 0.25))

    # Add granular stereo pad cloud for extra warmth
    if shared_pad is not None and shared_pad.shape[1] == total_samples:
        pad_stereo = shared_pad
    else:
        pad_stereo = create_granular_pad(
            prep.mono,
            target_duration=total_duration,
            semitone_shift=0.0,
            grain_duration=0.16,
            density=28.0,
            sr=sr,
            stereo_spread=True
        )
    chords_combined = resonated_stereo * 0.70 + pad_stereo * 0.45

    for sec in arrangement.sections:
        if "chords" in sec.active_layers:
            s_start = int(sec.start_time * sr)
            s_end = min(total_samples, int((sec.start_time + sec.duration) * sr))
            if s_end > s_start:
                c_gain = 0.75 * sec.energy_level
                track_chords[:, s_start:s_end] = chords_combined[:, s_start:s_end] * c_gain

    # --- STEM 3: PUNCHY DRUMS & PERCUSSION (Hybrid Source Kick, Snare, Hats, Foley) ---
    kick_times: List[int] = []  # For sidechain ducking on the atmospheric bed

    if beat_preference != "none":
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
                            kick_snd = render_noise_kick(k_slice, velocity=k_vel * 0.95, sr=sr)
                            kl = min(len(kick_snd), total_samples - step_s)
                            track_drums[0, step_s:step_s + kl] += kick_snd[:kl]
                            track_drums[1, step_s:step_s + kl] += kick_snd[:kl]
                            kick_times.append(step_s)

                        # 2. Snare / Clap
                        s_vel = rhythm_track.snare_pattern[step_idx]
                        if s_vel > 0:
                            snare_snd = render_noise_snare(s_slice, velocity=s_vel * 0.88, sr=sr)
                            sl = min(len(snare_snd), total_samples - step_s)
                            track_drums[0, step_s:step_s + sl] += snare_snd[:sl]
                            track_drums[1, step_s:step_s + sl] += snare_snd[:sl]

                        # 3. Hi-Hat / Shaker
                        h_vel = rhythm_track.hihat_pattern[step_idx]
                        if h_vel > 0:
                            hihat_snd = render_noise_hihat(h_slice, velocity=h_vel * 0.72, sr=sr)
                            hl = min(len(hihat_snd), total_samples - step_s)
                            pan_h = rng.uniform(-0.25, 0.25)
                            panned_h = apply_panning(hihat_snd[:hl], pan=pan_h)
                            track_drums[:, step_s:step_s + hl] += panned_h

                        # 4. Source Foley Percussion Chops
                        sp_vel = rhythm_track.source_perc_pattern[step_idx]
                        if sp_vel > 0 and palette.impacts:
                            foley_sl = palette.impacts[(step + b) % len(palette.impacts)].audio
                            fl = min(len(foley_sl), total_samples - step_s, int(0.18 * sr))
                            if fl > 0:
                                pan_f = rng.uniform(-0.45, 0.45)
                                panned_f = apply_panning(foley_sl[:fl] * sp_vel * 0.65, pan=pan_f)
                                track_drums[:, step_s:step_s + fl] += panned_f

            sec_bar_offset += sec.bars

    # Sidechain Ducking: Duck the ambient bed slightly on each kick hit
    duck_len = int(0.12 * sr)
    duck_curve = 1.0 - 0.40 * np.exp(-np.linspace(0, 4, duck_len))
    for ks in kick_times:
        d_end = min(total_samples, ks + duck_len)
        dl = d_end - ks
        if dl > 0:
            track_bed[:, ks:d_end] *= duck_curve[:dl]

    # --- STEM 4: DEEP GROOVING SUB-BASS (Fused with source low-end) ---
    for sec in arrangement.sections:
        if "bass" in sec.active_layers:
            for b in range(sec.bars):
                b_time = sec.start_time + b * arrangement.seconds_per_bar
                b_samp = int(b_time * sr)
                chord_idx = b % len(chords)
                chord = chords[chord_idx]
                bass_dur = arrangement.seconds_per_bar * 0.94

                rendered_bass = render_noise_bass_note(
                    source_audio=prep.mono,
                    freq=midi_to_hz(chord.root_midi),
                    duration=bass_dur,
                    velocity=0.90 * sec.energy_level,
                    sr=sr
                )
                bl = min(len(rendered_bass), total_samples - b_samp)
                if bl > 0 and b_samp < total_samples:
                    track_bass[0, b_samp:b_samp + bl] += rendered_bass[:bl]
                    track_bass[1, b_samp:b_samp + bl] += rendered_bass[:bl]

    # --- STEM 5: CATCHY MELODIC LEAD / HOOK (Tuned source instrument) ---
    lead_style = rng.choice(["pluck", "chime", "lead"])
    for event in melody_events:
        sec = next((s for s in arrangement.sections if s.start_time <= event.start_time < s.start_time + s.duration), None)
        if sec and "melody" in sec.active_layers:
            start_s = int(event.start_time * sr)
            if start_s < total_samples:
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

    # --- STEM 6: STRUCTURAL ACCENTS & TRANSITIONAL SWEEPS ---
    for sec in arrangement.sections:
        if "accents" in sec.active_layers:
            acc_slice = rng.choice(palette.accents)
            acc_s = int(sec.start_time * sr)
            al = min(len(acc_slice.audio), total_samples - acc_s)
            if al > 0:
                panned_acc = apply_panning(acc_slice.audio[:al] * 0.85, pan=rng.uniform(-0.3, 0.3))
                track_accents[:, acc_s:acc_s + al] += panned_acc

    # 8. Spatial Effects & Reverb Space
    track_melody = ping_pong_delay(track_melody, bpm=bpm, division=0.5, feedback=0.25, mix=0.22, sr=sr)
    track_melody = schroeder_reverb(track_melody, room_size=0.75, wet_level=0.22, sr=sr)
    track_chords = schroeder_reverb(track_chords, room_size=0.85, wet_level=0.28, sr=sr)
    # Bass drive tailored to genre
    bass_drive = 1.15
    if pref in ["rap", "trap", "drill"]:
        bass_drive = 1.42   # Saturated 808 sub-bass wall
    elif pref in ["hiphop", "hip_hop", "lofi"]:
        bass_drive = 1.25   # Warm thick boom-bap bass
    elif pref in ["pop"]:
        bass_drive = 1.10   # Clean punchy dance bass
    track_bass = soft_saturation(track_bass, drive=bass_drive)

    # 9. Professional Studio Mix Balancing
    mix = (
        track_bed * 0.45 +        # Original recording breathing underneath
        track_drums * 0.95 +      # Punchy kick, snare, hats upfront
        track_bass * 0.82 +       # Solid warm sub-bass groove
        track_chords * 0.72 +     # Lush harmonic progression
        track_melody * 0.78 +     # Catchy melodic hook
        track_accents * 0.55      # Dynamic transitional sweeps
    )

    # 10. Clean Mastering Chain (LUFS -14.0, True Peak Limiter -0.5 dB)
    mastered = master_audio(mix, target_lufs=-14.0, target_peak_db=-0.5, sr=sr)

    # 11. Objective Quality Scoring
    score = score_composition(
        output_audio=mastered,
        source_audio=prep.mono,
        source_usage_ratio=1.00,
        synthetic_audio_ratio=0.00,
        sr=sr
    )

    # Contextual stem descriptions reflecting genre mode
    if pref in ["hiphop", "hip_hop", "boom_bap", "lofi"]:
        drum_desc = "Boom-Bap Kick, Cracking Snare, Swung Hi-Hats & Vinyl Chops (Hip Hop Mode)"
        bass_desc = "Warm Syncopated Sub-Bass (Hip Hop 808/Acoustic)"
        lead_desc = f"Soulful Lead Hook & Chops ({lead_style.title()})"
    elif pref in ["rap", "trap", "drill"]:
        drum_desc = "Heavy 808 Kick, Rolling Sizzle Hats & Half-Time Claps (Rap Mode)"
        bass_desc = "Hard Saturated 808 Sub-Bass Wall (Rap Mode)"
        lead_desc = f"Dark Hypnotic Staccato Motif ({lead_style.title()})"
    elif pref in ["pop", "dance", "synthpop"]:
        drum_desc = "Punchy 4-on-the-Floor Kick, Pop Claps & Disco Open Hats (Pop Mode)"
        bass_desc = "Bouncy Driving Melodic Bassline (Pop Mode)"
        lead_desc = f"Catchy Earworm Melodic Hook ({lead_style.title()})"
    else:
        drum_desc = f"Source-Crafted Punchy Kick, Snare & Hi-Hats ({beat_preference.title()})"
        bass_desc = "Source-Textured Warm Sub-Bass"
        lead_desc = f"Tuned Source Melodic Hook ({lead_style.title()})"

    stems_info = {
        "Drums": drum_desc,
        "Bass": bass_desc,
        "Chords": f"Resonant Harmonic Progression ({scale.name})",
        "Melody": lead_desc,
        "Bed": "Atmospheric Source Recording Bed",
        "Accents": "Dynamic Foley Sweeps & Accents"
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
    beat_preference: str = "minimal",
    energy_preference: str = "balanced",
    num_candidates: int = 2,
    on_progress: Optional[Any] = None
) -> Tuple[CompositionResult, List[CompositionResult]]:
    """Generates candidates with independent seeds, ranked by score."""
    if base_seed is None:
        base_seed = generate_seed()

    # Pre-extract palette once and reuse across all candidates (saves redundant CPU/memory)
    palette = build_source_palette(
        audio=prep.mono,
        sr=prep.sr,
        onset_samples=analysis.rhythm.onset_samples,
        target_slice_count=48
    )

    # Pre-render shared pad once across candidates to avoid repeated heavy granular synthesis
    shared_pad = create_granular_pad(
        prep.mono,
        target_duration=60.0,
        semitone_shift=0.0,
        grain_duration=0.16,
        density=20.0,
        sr=prep.sr,
        stereo_spread=True
    )

    seeds = [base_seed]
    for _ in range(num_candidates - 1):
        seeds.append(generate_seed())

    import gc
    candidates: List[CompositionResult] = []
    for idx, s in enumerate(seeds):
        if on_progress:
            pct = 65 + int((idx / max(1, num_candidates)) * 20)
            on_progress(pct, f"Procedurally synthesizing candidate {idx + 1} of {num_candidates}...")
        cand = render_candidate_composition(
            prep=prep,
            analysis=analysis,
            seed=s,
            beat_preference=beat_preference,
            energy_preference=energy_preference,
            palette=palette,
            shared_pad=shared_pad
        )
        candidates.append(cand)
        gc.collect()
        time.sleep(0.01)  # Yield CPU slice to OS scheduler to prevent UI stutter

    candidates.sort(key=lambda c: c.score.total_score, reverse=True)
    return candidates[0], candidates
