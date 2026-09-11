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
    pref = (style_preference or "minimal").lower()
    if pref in ["hiphop", "hip_hop", "boom_bap", "boombap", "lofi"]:
        groove_type = "hiphop"
    elif pref in ["rap", "trap", "drill"]:
        groove_type = "rap"
    elif pref in ["pop", "dance", "synthpop"]:
        groove_type = "pop"
    elif pref in ["energetic"] or bpm >= 118.0:
        groove_type = "four_on_floor"
    elif bpm <= 86.0 or pref == "minimal":
        groove_type = "halftime"
    else:
        groove_type = "standard_boombap"

    base_kick = [0.0] * 16
    base_snare = [0.0] * 16
    base_hihat = [0.0] * 16
    base_foley = [0.0] * 16

    # --- 1. HIP HOP (Classic 90s Boom-Bap & Lo-Fi Swung Pocket) ---
    if groove_type == "hiphop":
        # Heavy punchy kick on 0 (beat 1)
        base_kick[0] = 0.98
        # Syncopated kick on step 10 (beat 3 upbeat)
        base_kick[10] = 0.90
        if rng.chance(0.5):
            base_kick[3] = 0.70  # Ghost kick
        if rng.chance(0.35):
            base_kick[8] = 0.82  # Beat 3 downbeat kick

        # Cracking snare on beats 2 & 4 (steps 4 and 12)
        base_snare[4] = 0.94
        base_snare[12] = 0.96
        # Ghost snare taps for authentic human pocket
        if rng.chance(0.4):
            base_snare[7] = 0.38
        if rng.chance(0.35):
            base_snare[15] = 0.45

        # Swung 8th-note hi-hats with dynamic velocity pocket
        for s in range(0, 16, 2):
            vel = 0.72 + (0.16 if s % 4 == 0 else -0.12) + rng.uniform(-0.05, 0.05)
            base_hihat[s] = float(np.clip(vel, 0.35, 0.90))
        # Swung off-beat 16th taps
        for s in [3, 7, 11, 15]:
            if rng.chance(0.6):
                base_hihat[s] = float(rng.uniform(0.35, 0.55))

        # Syncopated source vinyl chops & percussion
        for s in [2, 6, 9, 14]:
            if rng.chance(0.65):
                base_foley[s] = float(rng.uniform(0.55, 0.85))

    # --- 2. RAP / TRAP (Hard 808 Kick Bounces, Half-Time Claps & Sizzling Rolling Hats) ---
    elif groove_type == "rap":
        # Hard-hitting 808 kick pattern with iconic trap syncopation
        base_kick[0] = 0.98
        if rng.chance(0.75):
            base_kick[6] = 0.88   # Mid-measure double kick
        if rng.chance(0.85):
            base_kick[11] = 0.92  # Late syncopated kick
        if rng.chance(0.45):
            base_kick[14] = 0.84  # Pickup kick

        # Hard half-time trap snare/clap on beat 3 (step 8)
        base_snare[8] = 0.98

        # Sizzling rolling trap hi-hats across 16th-notes with velocity dynamics
        for s in range(16):
            base_hihat[s] = float(0.52 + (0.24 if s % 4 == 0 else 0.0) + rng.uniform(-0.06, 0.06))

        # Fast trap hat rolls on steps 6, 7, 14, 15
        base_hihat[6] = 0.78
        base_hihat[7] = 0.88
        base_hihat[14] = 0.84
        base_hihat[15] = 0.94

        # Syncopated 808 rimshots and vocal/source chops
        for s in [3, 9, 13]:
            if rng.chance(0.7):
                base_foley[s] = float(rng.uniform(0.65, 0.92))

    # --- 3. POP (Four-on-the-Floor Drive, Bright Claps & Disco Open Hats) ---
    elif groove_type == "pop":
        # Unstoppable four-on-the-floor kick pulse
        for s in [0, 4, 8, 12]:
            base_kick[s] = 0.96
        # Energetic pickup kick
        if rng.chance(0.5):
            base_kick[14] = 0.82

        # Bright pop snares/claps on beats 2 & 4 (steps 4 and 12)
        base_snare[4] = 0.94
        base_snare[12] = 0.96

        # Driving 16th-note hi-hats
        for s in range(16):
            base_hihat[s] = float(0.48 + (0.20 if s % 2 == 0 else 0.0))
        # Off-beat open hats (steps 2, 6, 10, 14)
        for s in [2, 6, 10, 14]:
            base_hihat[s] = 0.86

        # Tambourine & shaker source foley on 16th upbeats
        for s in range(16):
            if s % 2 == 1:
                base_foley[s] = float(rng.uniform(0.50, 0.78))

    # --- 4. FOUR-ON-THE-FLOOR (Energetic Dance) ---
    elif groove_type == "four_on_floor":
        for s in [0, 4, 8, 12]:
            base_kick[s] = 0.95
        for s in [4, 12]:
            base_snare[s] = 0.88
        for s in [2, 6, 10, 14]:
            base_hihat[s] = 0.75
        for s in [3, 7, 11, 15]:
            base_foley[s] = 0.55 if rng.chance(0.6) else 0.0

    # --- 5. HALF-TIME (Minimal / Downtempo) ---
    elif groove_type == "halftime":
        base_kick[0] = 0.95
        if rng.chance(0.7):
            base_kick[6] = 0.82
        if rng.chance(0.5):
            base_kick[10] = 0.78
        base_snare[8] = 0.92
        for s in range(0, 16, 2):
            vel = 0.50 + (0.25 if s % 4 == 0 else 0.0) + rng.uniform(-0.08, 0.08)
            base_hihat[s] = float(np.clip(vel, 0.3, 0.85))
        for s in [3, 7, 11, 14]:
            if rng.chance(0.5):
                base_foley[s] = rng.uniform(0.45, 0.75)

    # --- 6. STANDARD BOOM-BAP ---
    else:
        base_kick[0] = 0.96
        k2 = rng.choice([8, 10, 11])
        base_kick[k2] = 0.88
        if rng.chance(0.4):
            base_kick[3] = 0.70
        base_snare[4] = 0.90
        base_snare[12] = 0.92
        for s in range(0, 16, 2):
            vel = 0.55 + (0.20 if s % 4 == 0 else 0.0) + rng.uniform(-0.06, 0.06)
            base_hihat[s] = float(np.clip(vel, 0.35, 0.88))
        for s in [3, 7, 11, 15]:
            if rng.chance(0.35):
                base_hihat[s] = 0.40
        for s in [2, 6, 9, 14]:
            if rng.chance(0.45):
                base_foley[s] = rng.uniform(0.5, 0.8)

    # Tile across 4 bars with musical variations, fills, and drops
    for bar in range(4):
        offset = bar * 16
        for s in range(16):
            # Kick
            k_val = base_kick[s]
            if bar == 2 and s == 14 and rng.chance(0.6):
                k_val = 0.85  # Pickup kick before bar 4
            if bar == 3 and s == 14 and rng.chance(0.75):
                k_val = 0.90  # Turnaround kick
            # Drop the kick on bar 3, beat 4 in rap mode for dramatic tension
            if groove_type == "rap" and bar == 3 and s in [12, 13, 14, 15]:
                k_val = 0.0
            kick[offset + s] = k_val

            # Snare
            sn_val = base_snare[s]
            if groove_type == "rap" and bar == 3 and s in [12, 13, 14, 15]:
                # Rapid trap snare roll drop on bar 4 turnaround!
                sn_val = float(0.65 + (s - 12) * 0.10)
            elif bar == 3 and s in [14, 15] and rng.chance(0.6):
                sn_val = 0.80  # Snare roll at end of phrase
            snare[offset + s] = sn_val

            # Hi-Hat
            hh_val = base_hihat[s]
            if bar == 1 and s % 2 == 1 and rng.chance(0.3):
                hh_val = 0.50
            if groove_type == "rap" and bar == 3 and s >= 12:
                # 32nd-note burst roll feel
                hh_val = 0.88
            hihat[offset + s] = hh_val

            # Source Percussion
            sp_val = base_foley[s]
            if bar == 3 and s in [13, 15]:
                sp_val = 0.75  # Source chop turnaround
            source_perc[offset + s] = sp_val

    return RhythmTrack(
        style=groove_type,
        steps_per_bar=16,
        kick_pattern=kick,
        snare_pattern=snare,
        hihat_pattern=hihat,
        source_perc_pattern=source_perc
    )
