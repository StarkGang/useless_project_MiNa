"""
Procedural Musical Arrangement and Adaptive Form Engine
TheUnnecessaryFM Philosophy:
  - Build music through arrangement, dynamics, and silence, not synthetic instruments.
  - No rigid song template. Adapts form to the sonic characteristics of the recording.
  - Generates constrained structures targeting ~60 seconds (55-65s range).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from ..utils.random import SeededRNG


@dataclass
class Section:
    name: str                         # E.g. "Intro_Atmosphere", "Motif_Entry", "Full_Development", "Peak", "Breakdown", "Resolution"
    start_time: float                 # Seconds
    duration: float                   # Seconds
    bars: int                         # Number of bars
    energy_level: float               # Target intensity [0.0, 1.0]
    active_layers: List[str]          # "bed", "movement", "groove", "accents", "resonance"
    filter_cutoff: float              # Master/bus filter brightness in Hz


@dataclass
class CompositionArrangement:
    tempo_bpm: float
    total_bars: int
    seconds_per_bar: float
    total_duration: float             # In seconds (target 55-65s)
    sections: List[Section]
    selected_form: str
    available_instruments: List[str]


def plan_arrangement(
    bpm: float,
    source_energy_envelope: Optional[List[float]] = None,
    sonic_character: str = "MIXED",
    rng: Optional[SeededRNG] = None,
    target_sec: float = 30.0
) -> CompositionArrangement:
    """
    Plans a musical composition targeting the requested duration (30s or 60s).
    30s: tight drop → climax structure.
    60s: extended intro + full development + outro.
    """
    if rng is None:
        rng = SeededRNG()

    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * 4.0

    # Calculate total bars to match target duration
    ideal_bars = int(round(target_sec / seconds_per_bar))
    if ideal_bars % 2 != 0:
        ideal_bars += 1
    # Minimum bars: 8 for 30s, 16 for 60s
    min_bars = 16 if target_sec >= 50.0 else 8
    ideal_bars = max(min_bars, ideal_bars)

    total_duration = ideal_bars * seconds_per_bar
    # Allow ±20% of target duration
    low_bound = target_sec * 0.80
    high_bound = target_sec * 1.20
    if total_duration < low_bound:
        ideal_bars += 2
        total_duration = ideal_bars * seconds_per_bar
    elif total_duration > high_bound and ideal_bars > min_bars:
        ideal_bars -= 2
        total_duration = ideal_bars * seconds_per_bar

    # Dynamic 5-Part Song Structure:
    # Intro_Reveal -> Groove_Drop -> Build_Up -> Chorus_Climax -> Outro_Resolution
    form = ["Intro_Reveal", "Groove_Drop", "Build_Up", "Chorus_Climax", "Outro_Resolution"]
    bar_distribution = [0.15, 0.28, 0.15, 0.30, 0.12]

    num_sections = len(form)
    raw_bars = [max(1, int(round(d * ideal_bars))) for d in bar_distribution]
    # Rebalance sum to match ideal_bars exactly
    diff = ideal_bars - sum(raw_bars)
    idx = 1
    while diff != 0:
        if diff > 0:
            raw_bars[idx % num_sections] += 1
            diff -= 1
        else:
            if raw_bars[idx % num_sections] > 1:
                raw_bars[idx % num_sections] -= 1
                diff += 1
        idx += 1

    env = source_energy_envelope if source_energy_envelope else [0.5] * 32
    sections: List[Section] = []
    current_time = 0.0

    all_layers = ["bed", "drums", "bass", "chords", "melody", "accents", "riser"]

    for i, name in enumerate(form):
        bars = raw_bars[i]
        sec_dur = bars * seconds_per_bar

        progress_idx = int(round((current_time / total_duration) * (len(env) - 1)))
        progress_idx = min(len(env) - 1, max(0, progress_idx))
        source_intensity = env[progress_idx]

        if name == "Intro_Reveal":
            # 1. Clear exposure of the original source noise + subtle resonant harmony
            energy = min(0.65, max(0.40, source_intensity * 0.75))
            layers = ["bed", "chords"]
            cutoff = 6000.0
        elif name == "Groove_Drop":
            # 2. Main beat drops with punchy noise drums, bassline & foley groove
            energy = min(0.85, max(0.70, source_intensity * 1.0))
            layers = ["bed", "drums", "bass", "chords"]
            cutoff = 10000.0
        elif name == "Build_Up":
            # 3. Tension rises: filtered snare roll + noise riser sweep
            energy = min(0.88, max(0.68, source_intensity * 1.05))
            layers = ["bed", "drums", "chords", "riser"]
            cutoff = 13000.0
        elif name == "Chorus_Climax":
            # 4. Maximum musical peak! All stems active with soaring Karplus-Strong lead
            energy = min(1.0, max(0.88, source_intensity * 1.25))
            layers = ["bed", "drums", "bass", "chords", "melody", "accents"]
            cutoff = 18000.0
        else:  # Outro_Resolution
            # 5. Satisfying cadence resolving on tonic chord, fading into raw noise
            energy = min(0.70, max(0.45, source_intensity * 0.8))
            layers = ["bed", "chords", "bass", "accents"]
            cutoff = 8000.0

        sections.append(Section(
            name=name,
            start_time=round(current_time, 3),
            duration=round(sec_dur, 3),
            bars=bars,
            energy_level=round(float(energy), 2),
            active_layers=layers,
            filter_cutoff=cutoff
        ))
        current_time += sec_dur

    return CompositionArrangement(
        tempo_bpm=bpm,
        total_bars=ideal_bars,
        seconds_per_bar=round(seconds_per_bar, 3),
        total_duration=round(total_duration, 2),
        sections=sections,
        selected_form=" -> ".join(form),
        available_instruments=all_layers
    )
