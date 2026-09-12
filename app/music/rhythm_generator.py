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
    humanize_timing_ms: float = 0.0


def generate_rhythm(
    style_preference: str = "pop",
    bpm: float = 122.0,
    has_source_rhythm: bool = False,
    rng: Optional[SeededRNG] = None,
    artist_profile: Optional[object] = None
) -> RhythmTrack:
    """
    Generates cohesive, punchy drum patterns with authentic drum fills,
    swung open hi-hats, artist-specific groove mechanics, and reactive source foley syncopation.
    """
    if rng is None:
        rng = SeededRNG()

    if style_preference == "none" or (artist_profile and getattr(artist_profile, "groove_type", "") == "ambient_still"):
        return RhythmTrack(
            style="none",
            steps_per_bar=16,
            kick_pattern=[0.0] * 64,
            snare_pattern=[0.0] * 64,
            hihat_pattern=[0.0] * 64,
            source_perc_pattern=[0.0] * 64,
            open_hihat_pattern=[0.0] * 64,
            humanize_timing_ms=0.0
        )

    total_steps = 64
    kick = [0.0] * total_steps
    snare = [0.0] * total_steps
    hihat = [0.0] * total_steps
    open_hihat = [0.0] * total_steps
    source_perc = [0.0] * total_steps

    # Determine groove archetype based on artist profile or genre/tempo
    humanize_ms = 0.0
    if artist_profile is not None:
        groove_type = getattr(artist_profile, "groove_type", "pop")
        humanize_ms = float(getattr(artist_profile, "humanize_timing_ms", 0.0))
    else:
        pref = (style_preference or "pop").lower()
        if pref in ["trap", "drill"]:
            groove_type = "trap"
        elif pref in ["pop", "dance", "synthpop"]:
            groove_type = "pop"
        elif pref in ["hiphop", "hip_hop", "boom_bap", "boombap", "lofi"]:
            groove_type = "hiphop"
        elif pref in ["rap", "electro", "french_touch"]:
            groove_type = "rap"
        elif pref in ["minimal", "house", "garage"]:
            groove_type = "minimal"
        elif pref in ["rhythmic", "energetic", "light_percussion"]:
            groove_type = "rhythmic"
        elif bpm <= 86.0:
            groove_type = "hiphop"
        else:
            groove_type = "pop"

    base_kick = [0.0] * 16
    base_snare = [0.0] * 16
    base_hihat = [0.0] * 16
    base_open_hihat = [0.0] * 16
    base_foley = [0.0] * 16

    # ── 1. USHER / LIL JON CRUNK (Heavy Stomp, Rigid Straight 16ths) ──────────
    if groove_type == "crunk":
        for s in [0, 4, 8, 12]:
            base_kick[s] = 0.98  # Heavy sidechain stomp
        base_snare[4] = 0.98  # Layered snare + snap
        base_snare[12] = 0.98
        # Straight energetic 16ths at ~85% velocity, no swing
        for s in range(16):
            base_hihat[s] = 0.85
        base_open_hihat[2] = 0.80
        base_open_hihat[10] = 0.80
        for s in [3, 7, 11, 15]:
            base_foley[s] = float(rng.uniform(0.70, 0.90))

    # ── 2. MICHAEL JACKSON & QUINCY JONES (Syncopated Funk Pocket) ────────────
    elif groove_type == "funk_pocket":
        # Syncopated kick: not every downbeat
        base_kick[0] = 0.96
        base_kick[6] = 0.88
        base_kick[10] = 0.90
        if rng.chance(0.5):
            base_kick[14] = 0.80
        base_snare[4] = 0.96
        base_snare[12] = 0.98
        # 16th ghost-note snare taps at low velocity
        for s in [3, 7, 11, 15]:
            base_snare[s] = float(rng.uniform(0.28, 0.38))
        for s in range(16):
            base_hihat[s] = float(0.68 if s % 2 == 0 else 0.44)
        base_open_hihat[6] = 0.75
        base_open_hihat[14] = 0.78
        for s in [2, 8, 14]:
            base_foley[s] = float(rng.uniform(0.60, 0.82))

    # ── 3. THE WEEKND & MAX MARTIN (80s Driving Synthwave & Gated Snare) ──────
    elif groove_type == "synthwave_driving":
        # Driving straight 8th kick
        for s in range(0, 16, 2):
            base_kick[s] = 0.96
        # Big 80s gated snare on 2 & 4
        base_snare[4] = 0.98
        base_snare[12] = 0.98
        for s in range(16):
            base_hihat[s] = float(0.72 if s % 2 == 0 else 0.50)
        base_open_hihat[14] = 0.80
        for s in [3, 7, 11, 15]:
            base_foley[s] = float(rng.uniform(0.65, 0.85))

    # ── 4. DUA LIPA (Nu-Disco Boots-and-Cats) ──────────────────────────────────
    elif groove_type == "disco_boots_cats":
        for s in [0, 4, 8, 12]:
            base_kick[s] = 0.98  # Boots
        base_snare[4] = 0.96   # Claps
        base_snare[12] = 0.96
        # Cats: Persistent open hat on every single upbeat (2, 6, 10, 14)
        for s in [2, 6, 10, 14]:
            base_open_hihat[s] = 0.88
        for s in range(16):
            base_hihat[s] = float(0.55 if s % 2 == 0 else 0.40)
        for s in [1, 5, 9, 13]:
            base_foley[s] = float(rng.uniform(0.60, 0.80))

    # ── 5. BRUNO MARS (Live-Feel Funk Groove) ──────────────────────────────────
    elif groove_type == "live_funk_swung":
        base_kick[0] = 0.96
        base_kick[5] = 0.85
        base_kick[10] = 0.90
        base_snare[4] = 0.96
        base_snare[12] = 0.98
        # Ghost notes
        for s in [2, 7, 14]:
            base_snare[s] = float(rng.uniform(0.30, 0.42))
        for s in range(16):
            base_hihat[s] = float(0.65 + (0.15 if s % 4 == 0 else -0.10) + rng.uniform(-0.06, 0.06))
        base_open_hihat[10] = 0.78
        for s in [3, 9, 15]:
            base_foley[s] = float(rng.uniform(0.68, 0.88))

    # ── 6. DJ MUSTARD (Ultra-Sparse West Coast Ratchet Pocket) ─────────────────
    elif groove_type == "mustard_ratchet":
        # Ultra-sparse: silence is an instrument
        base_kick[0] = 0.98
        if rng.chance(0.4):
            base_kick[10] = 0.88
        base_snare[4] = 0.98  # Piercing mustard clap
        base_snare[12] = 0.98
        # Very sparse closed hats
        for s in [2, 6, 10, 14]:
            base_hihat[s] = 0.65
        base_open_hihat[14] = 0.75
        # Iconic vocal shout placement
        base_foley[4] = 0.90

    # ── 7. PHARRELL & THE NEPTUNES (Dry Minimal Funk) ─────────────────────────
    elif groove_type == "neptunes_dry":
        base_kick[0] = 0.96
        base_kick[6] = 0.86
        base_snare[4] = 0.96
        base_snare[12] = 0.96
        for s in [0, 4, 8, 12]:
            base_hihat[s] = 0.58
        for s in [2, 8, 14]:
            base_foley[s] = float(rng.uniform(0.70, 0.90))

    # ── 8. TIMBALAND (Asymmetric Foley Groove) ─────────────────────────────────
    elif groove_type == "timbaland_foley":
        base_kick[0] = 0.96
        base_kick[3] = 0.80
        base_kick[8] = 0.88
        base_kick[11] = 0.84
        base_snare[4] = 0.96
        base_snare[12] = 0.98
        for s in [2, 6, 9, 13]:
            base_hihat[s] = float(rng.uniform(0.55, 0.75))
        # Mouth clicks / vocal foley dominant
        for s in [1, 5, 7, 10, 14, 15]:
            base_foley[s] = float(rng.uniform(0.75, 0.95))

    # ── 9. METRO BOOMIN & ATLANTA TRAP ─────────────────────────────────────────
    elif groove_type in ["metro_halftime", "trap", "808mafia_drill", "pierre_bouncy", "murda_radio", "travis_psychedelic"]:
        base_snare[8] = 0.98  # Halftime beat 3
        if rng.chance(0.4):
            base_snare[15] = 0.45
        base_kick[0] = 0.98
        base_kick[6] = 0.88
        if rng.chance(0.6):
            base_kick[11] = 0.85
        if rng.chance(0.4):
            base_kick[14] = 0.78
        for s in range(16):
            base_hihat[s] = float(0.68 if s % 2 == 0 else 0.45)
        base_open_hihat[2] = 0.78
        base_open_hihat[10] = 0.82
        for s in [3, 7, 12, 14]:
            base_foley[s] = float(rng.uniform(0.60, 0.88))

    # ── 10. J DILLA / NUJABES / KANYE (Swung Boom-Bap) ─────────────────────────
    elif groove_type in ["dilla_lazy_swing", "soul_boombap", "nujabes_swung", "gfunk_pocket", "hiphop", "doom_swing"]:
        base_kick[0] = 0.98
        base_kick[10] = 0.90
        if rng.chance(0.5):
            base_kick[3] = 0.72
        if rng.chance(0.35):
            base_kick[8] = 0.82
        base_snare[4] = 0.94
        base_snare[12] = 0.96
        if rng.chance(0.4):
            base_snare[7] = 0.40
        if rng.chance(0.35):
            base_snare[15] = 0.45
        for s in range(0, 16, 2):
            vel = 0.72 + (0.16 if s % 4 == 0 else -0.12) + rng.uniform(-0.04, 0.04)
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

    # ── 11. DEMBOW / REGGAETON / MOOMBAHTON (3:2 Clave) ────────────────────────
    elif groove_type in ["dembow_clave", "reggaeton_dembow"]:
        for s in [0, 4, 8, 12]:
            base_kick[s] = 0.98  # Straight kick
        # Dembow snare hits on 3, 6, 11, 14
        for s in [3, 6, 11, 14]:
            base_snare[s] = 0.94
        for s in range(16):
            base_hihat[s] = float(0.60 if s % 2 == 0 else 0.40)
        base_open_hihat[14] = 0.75
        for s in [2, 7, 10]:
            base_foley[s] = float(rng.uniform(0.65, 0.85))

    # ── 12. MINIMAL / 2-STEP (Four Tet, Burial, Jamie xx) ─────────────────────
    elif groove_type in ["fourtet_skipping", "burial_2step", "jamiexx_shuffle", "bonobo_organic_groove", "aphex_acid_808", "minimal"]:
        base_kick[0] = 0.96
        base_kick[7] = 0.88
        if rng.chance(0.5):
            base_kick[10] = 0.75
        base_snare[4] = 0.92
        base_snare[12] = 0.94
        base_snare[9] = 0.40
        for s in [1, 3, 5, 7, 9, 11, 13, 15]:
            base_hihat[s] = float(rng.uniform(0.45, 0.75))
        base_hihat[2] = 0.50
        base_hihat[10] = 0.52
        base_open_hihat[14] = 0.70
        for s in [2, 5, 8, 11, 13, 14]:
            base_foley[s] = float(rng.uniform(0.65, 0.90))

    # ── 13. STANDARD POP / DANCE FALLBACK ─────────────────────────────────────
    else:
        for s in [0, 4, 8, 12]:
            base_kick[s] = 0.98
        base_snare[4] = 0.96
        base_snare[12] = 0.98
        for s in range(16):
            base_hihat[s] = float(0.70 if s % 2 == 0 else 0.48)
        for s in [2, 6, 10, 14]:
            base_open_hihat[s] = 0.85
        for s in [3, 7, 11, 14, 15]:
            base_foley[s] = float(rng.uniform(0.70, 0.92))

    # Tile across 4 bars with musical variations, authentic trap rolls, fills, and drops
    for bar in range(4):
        offset = bar * 16
        for s in range(16):
            # Kick
            k_val = base_kick[s]
            if bar == 2 and s == 14 and rng.chance(0.6):
                k_val = 0.85  # Pickup kick before bar 4
            if bar == 3 and s in [12, 14] and rng.chance(0.75):
                k_val = 0.90  # Turnaround kick
            # Drop the kick on bar 3, beat 4 in rap/trap mode for dramatic drop tension
            if ("trap" in groove_type or "rap" in groove_type) and bar == 3 and s in [12, 13, 14, 15]:
                k_val = 0.0
            kick[offset + s] = k_val

            # Snare
            sn_val = base_snare[s]
            if bar == 3 and s in [10, 12, 13, 14, 15]:
                sn_val = float(0.65 + (s - 10) * 0.07)
            elif bar == 1 and s == 15 and rng.chance(0.5):
                sn_val = 0.60
            snare[offset + s] = sn_val

            # Hi-Hat
            hh_val = base_hihat[s]
            if bar == 1 and s % 2 == 1 and rng.chance(0.3):
                hh_val = 0.52
            if "trap" in groove_type and bar in [1, 3] and s in [6, 7, 14, 15]:
                hh_val = 0.92  # High-velocity roll
            elif ("rap" in groove_type or "crunk" in groove_type) and bar == 3 and s >= 12:
                hh_val = 0.88
            hihat[offset + s] = hh_val

            # Open Hi-Hat
            oh_val = base_open_hihat[s]
            if bar == 3 and s == 14:
                oh_val = 0.85
            open_hihat[offset + s] = oh_val

            # Source Percussion
            sp_val = base_foley[s]
            if bar == 3 and s in [11, 13, 15]:
                sp_val = 0.88
            source_perc[offset + s] = sp_val

    return RhythmTrack(
        style=groove_type,
        steps_per_bar=16,
        kick_pattern=kick,
        snare_pattern=snare,
        hihat_pattern=hihat,
        open_hihat_pattern=open_hihat,
        source_perc_pattern=source_perc,
        humanize_timing_ms=humanize_ms
    )
