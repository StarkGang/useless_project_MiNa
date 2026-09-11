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
    rng: Optional[SeededRNG] = None
) -> CompositionArrangement:
    """
    Plans an adaptive musical arc strictly bounded between 55 and 65 seconds (~60s).
    """
    if rng is None:
        rng = SeededRNG()

    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * 4.0

    # Calculate total bars to stay tightly in 55-65s window
    target_sec = 60.0
    ideal_bars = int(round(target_sec / seconds_per_bar))
    if ideal_bars % 2 != 0:
        ideal_bars += 1

    total_duration = ideal_bars * seconds_per_bar
    if total_duration < 55.0:
        ideal_bars += 2
        total_duration = ideal_bars * seconds_per_bar
    elif total_duration > 65.0:
        ideal_bars -= 2
        total_duration = ideal_bars * seconds_per_bar

    # Select adaptive form template based on sonic character
    if sonic_character in ["AMBIENT", "TEXTURAL"]:
        # Evolving atmospheric journey
        form = ["Atmosphere", "First_Pulse", "Expanding_Space", "Peak_Resonance", "Deconstruct", "Resolution"]
        bar_distribution = [0.18, 0.18, 0.22, 0.20, 0.12, 0.10]
    elif sonic_character in ["PERCUSSIVE", "RHYTHMIC"]:
        # Dynamic groove arc with breakdown
        form = ["Source_Intro", "Groove_Entry", "Layered_Drive", "Peak_Syncopation", "Breakdown", "Source_Fade"]
        bar_distribution = [0.15, 0.20, 0.25, 0.20, 0.10, 0.10]
    else:
        # Standard dynamic narrative arc
        form = ["Introduction", "Motif_Entry", "Full_Development", "Peak_Energy", "Breathing_Room", "Resolution"]
        bar_distribution = [0.16, 0.18, 0.24, 0.22, 0.10, 0.10]

    num_sections = len(form)
    raw_bars = [max(2, int(round(d * ideal_bars))) for d in bar_distribution]
    # Rebalance sum to match ideal_bars exactly
    diff = ideal_bars - sum(raw_bars)
    idx = 2
    while diff != 0:
        if diff > 0:
            raw_bars[idx % num_sections] += 1
            diff -= 1
        else:
            if raw_bars[idx % num_sections] > 2:
                raw_bars[idx % num_sections] -= 1
                diff += 1
        idx += 1

    env = source_energy_envelope if source_energy_envelope else [0.5] * 32
    sections: List[Section] = []
    current_time = 0.0

    all_layers = ["bed", "drums", "bass", "chords", "melody", "accents"]

    for i, name in enumerate(form):
        bars = raw_bars[i]
        sec_dur = bars * seconds_per_bar

        progress_idx = int(round((current_time / total_duration) * (len(env) - 1)))
        progress_idx = min(len(env) - 1, max(0, progress_idx))
        source_intensity = env[progress_idx]

        if i == 0:
            # 1. Intro: Atmospheric source bed + harmonic chords swell
            energy = min(0.45, max(0.25, source_intensity * 0.7))
            layers = ["bed", "chords"]
            cutoff = 4000.0
        elif i == 1:
            # 2. Beat Drop / Groove Entry: Drums + Bass + Chords lock in!
            energy = min(0.70, max(0.50, source_intensity * 0.9))
            layers = ["bed", "drums", "bass", "chords"]
            cutoff = 8000.0
        elif i == 2:
            # 3. Full Development / Chorus: Melodic hook enters, accents punctuate
            energy = min(0.88, max(0.70, source_intensity * 1.05))
            layers = ["bed", "drums", "bass", "chords", "melody", "accents"]
            cutoff = 12000.0
        elif i == 3:
            # 4. Peak Energy: Full beat, soaring hook, intense drive
            energy = min(1.0, max(0.85, source_intensity * 1.2))
            layers = ["bed", "drums", "bass", "chords", "melody", "accents"]
            cutoff = 16000.0
        elif i == 4:
            # 5. Breakdown / Tension: Drums & bass drop out, emotional breathing space
            energy = min(0.40, source_intensity * 0.55)
            layers = ["bed", "chords", "melody"]
            cutoff = 3500.0
        else:
            # 6. Resolution: Soft chords resolve into the pure essence of the source recording
            energy = min(0.30, source_intensity * 0.45)
            layers = ["bed", "chords"]
            cutoff = 2500.0

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
