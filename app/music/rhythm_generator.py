"""
Procedural Rhythm and Beat Generation Engine
TheUnnecessaryFM Philosophy:
  - Punchy, infectious, head-nodding drum grooves.
  - Combines kick, snare, hi-hats, and syncopated source foley chops.
  - Generates 4-bar evolving patterns with turnaround fills on bar 4.
  - Authentic artist styles: Lady Gaga & Nelly Furtado for Pop, Kanye West & Daft Punk for Rap.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from ..utils.random import SeededRNG


@dataclass
class RhythmTrack:
    style: str                        # "pop", "rap", "hiphop", "four_on_floor", "halftime", "none"
    steps_per_bar: int                # 16 (for 16th notes)
    kick_pattern: List[float]         # Velocity array for 4-bar loop (64 steps)
    snare_pattern: List[float]
    hihat_pattern: List[float]
    source_perc_pattern: List[float]  # Syncopated foley chops from the uploaded sound
    open_hihat_pattern: Optional[List[float]] = None  # Sizzling open hi-hat accents


def generate_rhythm(
    style_preference: str = "pop",
    bpm: float = 122.0,
    has_source_rhythm: bool = False,
    rng: Optional[SeededRNG] = None
) -> RhythmTrack:
    """
    Generates cohesive, punchy drum patterns with authentic drum fills,
    swung open hi-hats, and reactive source foley syncopation.
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
            source_perc_pattern=[0.0] * 64,
            open_hihat_pattern=[0.0] * 64
        )

    total_steps = 64
    kick = [0.0] * total_steps
    snare = [0.0] * total_steps
    hihat = [0.0] * total_steps
    open_hihat = [0.0] * total_steps
    source_perc = [0.0] * total_steps

    # Determine groove archetype based on tempo and style
    pref = (style_preference or "pop").lower()
    if pref in ["pop", "dance", "synthpop"]:
        groove_type = "pop"
    elif pref in ["rap", "trap", "drill"]:
        groove_type = "rap"
    elif pref in ["hiphop", "hip_hop", "boom_bap", "boombap", "lofi"]:
        groove_type = "hiphop"
    elif pref in ["energetic"] or bpm >= 118.0:
        groove_type = "four_on_floor"
    elif bpm <= 86.0 or pref == "minimal":
        groove_type = "halftime"
    else:
        groove_type = "pop"

    base_kick = [0.0] * 16
    base_snare = [0.0] * 16
    base_hihat = [0.0] * 16
    base_open_hihat = [0.0] * 16
    base_foley = [0.0] * 16

    # --- 1. POP (Lady Gaga & Nelly Furtado — Four-on-the-Floor, Timbaland Stutters & Disco Open Hats) ---
    if groove_type == "pop":
        # Unstoppable four-on-the-floor kick pulse (Lady Gaga "Poker Face" / "Just Dance")
        for s in [0, 4, 8, 12]:
            base_kick[s] = 0.98

        # Bright punchy pop snares & layered claps on beats 2 & 4 (steps 4 and 12)
        base_snare[4] = 0.96
        base_snare[12] = 0.98

        # Driving 16th-note sizzle hi-hats
        for s in range(16):
            base_hihat[s] = float(0.70 if s % 2 == 0 else 0.48)

        # Classic disco open hi-hat on every single upbeat (steps 2, 6, 10, 14)
        for s in [2, 6, 10, 14]:
            base_open_hihat[s] = 0.85

        # Timbaland & RedOne signature: Syncopated stutter chops & foley from the user's noise!
        for s in [3, 7, 11, 14, 15]:
            base_foley[s] = float(rng.uniform(0.75, 0.95))

    # --- 2. RAP / HIP-HOP (Kanye West & Daft Punk — Electro-Hop Pocket & Soul/Foley Chops) ---
    elif groove_type == "rap":
        # Kanye x Daft Punk "Stronger" / French-touch electro-hip-hop punch
        base_kick[0] = 0.98
        base_kick[6] = 0.88   # Syncopated mid-bar punch
        base_kick[10] = 0.92  # Driving upbeat kick
        if rng.chance(0.5):
            base_kick[14] = 0.84  # Pickup kick

        # Chunky disco-hop clap & snare on beats 2 & 4 (steps 4 and 12)
        base_snare[4] = 0.96
        base_snare[12] = 0.98
        if rng.chance(0.4):
            base_snare[15] = 0.45

        # Daft Punk swung disco hi-hats
        for s in range(16):
            base_hihat[s] = float(0.56 + (0.22 if s % 2 == 0 else -0.10) + rng.uniform(-0.05, 0.05))

        base_open_hihat[2] = 0.82
        base_open_hihat[10] = 0.84

        # Kanye West Soul Chops & Daft Punk Vocal Stutters from the user's noise!
        for s in [2, 6, 8, 11, 14]:
            base_foley[s] = float(rng.uniform(0.78, 0.95))

    # --- 3. CLASSIC HIP HOP (J Dilla & Nujabes — Swung Pocket & Vinyl Foley) ---
    elif groove_type == "hiphop":
        base_kick[0] = 0.98
        base_kick[10] = 0.90
        if rng.chance(0.5):
            base_kick[3] = 0.70
        if rng.chance(0.35):
            base_kick[8] = 0.82

        base_snare[4] = 0.94
        base_snare[12] = 0.96
        if rng.chance(0.4):
            base_snare[7] = 0.38
        if rng.chance(0.35):
            base_snare[15] = 0.45

        for s in range(0, 16, 2):
            vel = 0.72 + (0.16 if s % 4 == 0 else -0.12) + rng.uniform(-0.05, 0.05)
            base_hihat[s] = float(np.clip(vel, 0.35, 0.90))
        for s in [3, 7, 11, 15]:
            if rng.chance(0.6):
                base_hihat[s] = float(rng.uniform(0.35, 0.55))

        if rng.chance(0.7):
            base_open_hihat[6] = 0.75
        if rng.chance(0.6):
            base_open_hihat[14] = 0.78

        for s in [2, 6, 9, 14]:
            if rng.chance(0.65):
                base_foley[s] = float(rng.uniform(0.55, 0.85))

    # --- 4. FOUR-ON-THE-FLOOR (Energetic Dance / Club) ---
    elif groove_type == "four_on_floor":
        for s in [0, 4, 8, 12]:
            base_kick[s] = 0.95
        base_snare[4] = 0.90
        base_snare[12] = 0.92
        for s in range(16):
            base_hihat[s] = 0.65 if s % 2 == 0 else 0.40
        for s in [2, 6, 10, 14]:
            base_open_hihat[s] = 0.70
        for s in [6, 14]:
            base_foley[s] = 0.70

    # --- 5. HALF-TIME (Minimal / Downtempo) ---
    else:
        base_kick[0] = 0.94
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
        base_open_hihat[14] = 0.68
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
            if bar == 3 and s in [12, 14] and rng.chance(0.75):
                k_val = 0.90  # Turnaround kick
            # Drop the kick on bar 3, beat 4 in rap mode for dramatic tension
            if groove_type == "rap" and bar == 3 and s in [12, 13, 14, 15]:
                k_val = 0.0
            kick[offset + s] = k_val

            # Snare
            sn_val = base_snare[s]
            if bar == 3 and s in [10, 12, 13, 14, 15]:
                # Dynamic drum fill on bar 4 turnaround!
                sn_val = float(0.65 + (s - 10) * 0.07)
            elif bar == 1 and s == 15 and rng.chance(0.5):
                sn_val = 0.60  # Ghost tap
            snare[offset + s] = sn_val

            # Hi-Hat
            hh_val = base_hihat[s]
            if bar == 1 and s % 2 == 1 and rng.chance(0.3):
                hh_val = 0.50
            if groove_type == "rap" and bar == 3 and s >= 12:
                # 32nd-note burst roll feel
                hh_val = 0.88
            hihat[offset + s] = hh_val

            # Open Hi-Hat
            oh_val = base_open_hihat[s]
            if bar == 3 and s == 14:
                oh_val = 0.85  # Accented open hat before drop
            open_hihat[offset + s] = oh_val

            # Source Percussion
            sp_val = base_foley[s]
            if bar == 3 and s in [11, 13, 15]:
                sp_val = 0.85  # Rhythmic foley rush into next bar
            source_perc[offset + s] = sp_val

    return RhythmTrack(
        style=groove_type,
        steps_per_bar=16,
        kick_pattern=kick,
        snare_pattern=snare,
        hihat_pattern=hihat,
        open_hihat_pattern=open_hihat,
        source_perc_pattern=source_perc
    )
