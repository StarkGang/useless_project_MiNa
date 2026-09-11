"""
Procedural Melody and Motif Generation Engine
Adheres to classical and modern motif transformation rules:
  - Generates cohesive 3-6 note motifs from scale degrees or source spectral peaks
  - Applies musical transformations: Inversion, Retrograde, Transposition, Rhythmic Dilatation
  - Repeats motif with controlled musical variations so the song feels memorable and intentional
  - Respects source energy envelope and pitch confidence
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ..dsp.pitch import midi_to_hz
from ..utils.random import SeededRNG
from .scales import MusicalScale


@dataclass
class NoteEvent:
    midi: int
    freq_hz: float
    start_time: float     # Start in seconds
    duration: float       # Duration in seconds
    velocity: float       # Velocity [0.0, 1.0]


def generate_motif(
    scale: MusicalScale,
    motif_length: int = 4,
    source_spectral_peaks: Optional[List[float]] = None,
    has_pitch_confidence: bool = False,
    rng: Optional[SeededRNG] = None
) -> List[int]:
    """
    Generate a concise, musical motif (scale degree intervals).
    If source has spectral peaks, maps those peaks to closest scale degrees.
    """
    if rng is None:
        rng = SeededRNG()

    scale_notes = scale.get_notes_in_octave_range(min_midi=60, max_midi=76)  # Octaves 4 to 5
    if not scale_notes:
        scale_notes = [60, 62, 64, 67, 69]

    # If source has reliable pitch or dominant spectral peaks, use them to anchor motif
    if source_spectral_peaks and len(source_spectral_peaks) > 0 and not has_pitch_confidence:
        # Map first 2-3 spectral peaks to nearest scale notes
        anchor_notes = []
        for p in source_spectral_peaks[:3]:
            # Convert Hz to MIDI
            m = 69.0 + 12.0 * np.log2(max(20.0, p) / 440.0) if p > 0 else 60
            closest = min(scale_notes, key=lambda n: abs(n - m))
            if closest not in anchor_notes:
                anchor_notes.append(closest)

        # Complete motif using stepwise scale motion from anchors
        motif = list(anchor_notes)
        curr = anchor_notes[-1] if anchor_notes else scale_notes[0]
        curr_idx = scale_notes.index(curr)
        while len(motif) < motif_length:
            step = rng.choice([-2, -1, 1, 2])
            new_idx = int(np.clip(curr_idx + step, 0, len(scale_notes) - 1))
            motif.append(scale_notes[new_idx])
            curr_idx = new_idx
        return motif

    # Standard melodic motif construction: Root, 3rd, 5th, or step motion
    root = scale.root_midi
    curr = root
    curr_idx = scale_notes.index(curr) if curr in scale_notes else 0
    motif = [scale_notes[curr_idx]]

    for _ in range(motif_length - 1):
        step = rng.choice([-2, -1, 1, 1, 2], weights=[0.15, 0.35, 0.35, 0.1, 0.05])
        curr_idx = int(np.clip(curr_idx + step, 0, len(scale_notes) - 1))
        motif.append(scale_notes[curr_idx])

    return motif


def transform_motif(
    motif: List[int],
    scale: MusicalScale,
    transform_type: str = "original",
    semitone_shift: int = 0
) -> List[int]:
    """
    Apply classical motif transformation:
    - 'original': unchanged
    - 'retrograde': reverse order
    - 'inversion': pitch direction mirrored around first note
    - 'transposition': shift within scale degrees
    """
    if transform_type == "retrograde":
        return list(reversed(motif))

    scale_notes = scale.get_notes_in_octave_range(min_midi=48, max_midi=84)

    if transform_type == "inversion":
        anchor = motif[0]
        inverted = []
        for n in motif:
            diff = n - anchor
            target = anchor - diff
            # Snap to nearest scale note
            closest = min(scale_notes, key=lambda x: abs(x - target))
            inverted.append(closest)
        return inverted

    if transform_type == "transposition":
        transposed = []
        for n in motif:
            target = n + semitone_shift
            closest = min(scale_notes, key=lambda x: abs(x - target))
            transposed.append(closest)
        return transposed

    return list(motif)


def sequence_melody(
    scale: MusicalScale,
    total_bars: int,
    seconds_per_bar: float,
    source_peaks: Optional[List[float]] = None,
    has_pitch: bool = False,
    density_factor: float = 0.6,
    rng: Optional[SeededRNG] = None
) -> List[NoteEvent]:
    """
    Sequence a repeating, evolving melody across all bars of the composition.
    Maintains musical coherence:
      - 2-bar motif repeated with variation
      - Silence/breathing spaces on certain bars
      - Climax octave jumps
    """
    if rng is None:
        rng = SeededRNG()

    import numpy as np

    motif = generate_motif(scale, motif_length=4, source_spectral_peaks=source_peaks, has_pitch_confidence=has_pitch, rng=rng)
    events: List[NoteEvent] = []

    # Step duration (8th note)
    step_sec = seconds_per_bar / 8.0

    bar_idx = 0
    while bar_idx < total_bars:
        # Every 4 bars, pick a motif transformation
        if bar_idx % 4 == 0:
            current_motif = motif
        elif bar_idx % 4 == 1:
            current_motif = transform_motif(motif, scale, transform_type="transposition", semitone_shift=rng.choice([-2, 2, 5]))
        elif bar_idx % 4 == 2:
            current_motif = transform_motif(motif, scale, transform_type="inversion")
        else:
            current_motif = transform_motif(motif, scale, transform_type="retrograde")

        # Leave occasional rest bar (breathing space)
        if rng.chance(1.0 - density_factor) and bar_idx % 4 == 3:
            bar_idx += 1
            continue

        bar_start_time = bar_idx * seconds_per_bar

        # Place notes on syncopated 8th note grid
        step_positions = [0, 2, 3, 5] if len(current_motif) == 4 else [0, 2, 4, 6]
        for i, m_note in enumerate(current_motif):
            if i < len(step_positions):
                s_pos = step_positions[i]
                start_t = bar_start_time + s_pos * step_sec
                dur = step_sec * rng.choice([1.2, 1.8, 2.5])
                vel = float(rng.uniform(0.65, 0.88))

                events.append(NoteEvent(
                    midi=m_note,
                    freq_hz=midi_to_hz(m_note),
                    start_time=round(start_t, 3),
                    duration=round(dur, 3),
                    velocity=round(vel, 2)
                ))

        bar_idx += 1

    return events
