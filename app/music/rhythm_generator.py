"""
Procedural Rhythm and Beat Generation Engine
TheUnnecessaryFM Philosophy:
  - Punchy, infectious, head-nodding drum grooves.
  - Combines kick, snare, hi-hats, and syncopated source foley chops.
  - Generates 4-bar evolving patterns with turnaround fills on bar 4.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from ..utils.random import SeededRNG


@dataclass
class RhythmTrack:
    style: str                        # "minimal", "lofi", "energetic", "driving", "ambient"
    steps_per_bar: int                # 16 (for 16th notes)
    kick_pattern: List[float]         # Velocity array for 4-bar loop (64 steps)
    snare_pattern: List[float]
    hihat_pattern: List[float]
    source_perc_pattern: List[float]  # Syncopated foley chops from the uploaded sound


def generate_rhythm(
    style_preference: str = "minimal",
    bpm: float = 95.0,
    has_source_rhythm: bool = False,
    rng: Optional[SeededRNG] = None
) -> RhythmTrack:
    """
    Generates cohesive, punchy drum patterns that lock in with the tempo and mood.
    """
    if rng is None:
        rng = SeededRNG()

    if style_preference == "none":
        return RhythmTrack(
            style="none",
            steps_per_bar=16,
            kick_pattern=[0.0] * 64,
            snare_pattern=[0.0] * 64,
            hihat_pattern=[0.0] * 64,
            source_perc_pattern=[0.0] * 64
        )

    total_steps = 64
    kick = [0.0] * total_steps
    snare = [0.0] * total_steps
    hihat = [0.0] * total_steps
    source_perc = [0.0] * total_steps

    # Determine groove archetype based on tempo and style
    # Archetype A: Boom-Bap / Downtempo (Kick on 0, 10; Snare on 4, 12)
    # Archetype B: Half-Time Lo-Fi (Kick on 0, late 6/8; Snare on 8)
    # Archetype C: Driving Four-on-the-Floor (Kick on 0, 4, 8, 12; Snare on 4, 12)
    if style_preference in ["energetic", "dance"] or bpm >= 118.0:
        groove_type = "four_on_floor"
    elif bpm <= 86.0 or style_preference == "minimal":
        groove_type = "halftime"
    else:
        groove_type = "standard_boombap"

    base_kick = [0.0] * 16
    base_snare = [0.0] * 16
    base_hihat = [0.0] * 16
    base_foley = [0.0] * 16

    if groove_type == "four_on_floor":
        for s in [0, 4, 8, 12]:
            base_kick[s] = 0.95
        for s in [4, 12]:
            base_snare[s] = 0.88
        for s in [2, 6, 10, 14]:
            base_hihat[s] = 0.75
        for s in [3, 7, 11, 15]:
            base_foley[s] = 0.55 if rng.chance(0.6) else 0.0

    elif groove_type == "halftime":
        base_kick[0] = 0.95
        if rng.chance(0.7):
            base_kick[6] = 0.82
        if rng.chance(0.5):
            base_kick[10] = 0.78
        # Heavy half-time snare on beat 3 (step 8)
        base_snare[8] = 0.92
        # Grooving 8th-note hi-hats with velocity dynamics
        for s in range(0, 16, 2):
            vel = 0.50 + (0.25 if s % 4 == 0 else 0.0) + rng.uniform(-0.08, 0.08)
            base_hihat[s] = float(np.clip(vel, 0.3, 0.85))
        # Syncopated foley ghost chops
        for s in [3, 7, 11, 14]:
            if rng.chance(0.5):
                base_foley[s] = rng.uniform(0.45, 0.75)

    else:  # standard_boombap
        base_kick[0] = 0.96
        # Syncopated 2nd kick
        k2 = rng.choice([8, 10, 11])
        base_kick[k2] = 0.88
        if rng.chance(0.4):
            base_kick[3] = 0.70

        # Snare on beats 2 & 4 (steps 4 and 12)
        base_snare[4] = 0.90
        base_snare[12] = 0.92

        # 8th note hi-hats
        for s in range(0, 16, 2):
            vel = 0.55 + (0.20 if s % 4 == 0 else 0.0) + rng.uniform(-0.06, 0.06)
            base_hihat[s] = float(np.clip(vel, 0.35, 0.88))

        # Ghost 16th hats
        for s in [3, 7, 11, 15]:
            if rng.chance(0.35):
                base_hihat[s] = 0.40

        # Syncopated source percussion
        for s in [2, 6, 9, 14]:
            if rng.chance(0.45):
                base_foley[s] = rng.uniform(0.5, 0.8)

    # Tile across 4 bars with musical variations and turnaround fill
    for bar in range(4):
        offset = bar * 16
        for s in range(16):
            # Kick
            k_val = base_kick[s]
            if bar == 2 and s == 14 and rng.chance(0.5):
                k_val = 0.80  # Pickup kick before bar 4
            if bar == 3 and s == 14 and rng.chance(0.7):
                k_val = 0.85  # Turnaround kick
            kick[offset + s] = k_val

            # Snare
            sn_val = base_snare[s]
            if bar == 3 and s in [14, 15] and rng.chance(0.5):
                sn_val = 0.75  # Snare roll at end of phrase
            snare[offset + s] = sn_val

            # Hi-Hat
            hh_val = base_hihat[s]
            if bar == 1 and s % 2 == 1 and rng.chance(0.3):
                hh_val = 0.45
            hihat[offset + s] = hh_val

            # Source Percussion
            sp_val = base_foley[s]
            if bar == 3 and s in [13, 15]:
                sp_val = 0.70  # Source chop turnaround
            source_perc[offset + s] = sp_val

    return RhythmTrack(
        style=groove_type,
        steps_per_bar=16,
        kick_pattern=kick,
        snare_pattern=snare,
        hihat_pattern=hihat,
        source_perc_pattern=source_perc
    )
