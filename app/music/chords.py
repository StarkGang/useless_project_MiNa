"""
Procedural Chord Progressions and Voicings
Generates diatonic and modal chord progressions (2, 3, 4, 5 chords) with proper voice leading.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from ..dsp.pitch import midi_to_hz
from ..utils.random import SeededRNG
from .scales import MusicalScale

# Standard progression degree patterns
PROGRESSIONS_MAJOR = [
    [1, 5, 6, 4],       # I - V - vi - IV
    [1, 6, 4, 5],       # I - vi - IV - V
    [1, 4, 5],          # I - IV - V
    [2, 5, 1],          # ii - V - I
    [1, 4, 6, 5],       # I - IV - vi - V
    [1, 5, 4],          # I - V - IV
    [1, 4],             # I - IV (Ambient 2-chord loop)
    [1, 6, 2, 5]        # I - vi - ii - V
]

PROGRESSIONS_MINOR = [
    [1, 6, 3, 7],       # i - VI - III - VII
    [1, 4, 6, 5],       # i - iv - VI - V
    [1, 7, 6],          # i - VII - VI
    [1, 4, 7],          # i - iv - VII
    [1, 6, 7],          # i - VI - VII
    [1, 4, 5],          # i - iv - v
    [1, 7],             # i - VII (Ambient modal loop)
    [1, 3, 4, 6]        # i - III - iv - VI
]


# Genre-specific progression degree patterns tailored to iconic artist archetypes
PROGRESSIONS_POP = [
    [6, 4, 1, 5],       # vi - IV - I - V (Lady Gaga "Poker Face", Nelly Furtado "Say It Right" anthem)
    [1, 5, 6, 4],       # I - V - vi - IV (Lady Gaga "Bad Romance" / RedOne iconic 4-chord progression)
    [1, 4, 6, 5],       # I - IV - vi - V (Driving synth-pop club progression)
    [6, 5, 4, 5],       # vi - V - IV - V (Nelly Furtado "Promiscuous" dramatic pop loop)
    [1, 6, 4, 5],       # I - vi - IV - V (Classic dance-pop hook)
]

PROGRESSIONS_RAP = [
    [1, 7, 6, 7],       # i - VII - VI - VII (French touch anthem)
    [1, 4, 1, 7],       # i - iv - i - VII (Funky disco vamp)
    [1, 6, 3, 7],       # i - VI - III - VII (Soulful hip hop progression)
    [1, 7],             # i - VII (Hypnotic 2-chord funk groove)
    [1, 6],             # i - VI (Driving minor tension)
]

PROGRESSIONS_HIPHOP = [
    [1, 6, 2, 5],       # i - VI - ii - V (Classic soulful jazz boom-bap)
    [1, 4, 2, 5],       # i - iv - ii - V (Lo-fi hip hop turnaround)
    [1, 7, 6, 5],       # i - VII - VI - V (Moody Andalusian cadence)
    [2, 5, 1, 6],       # ii - V - I - vi (Jazz hop classic)
    [1, 4, 5, 4],       # i - iv - v - iv (Mellow boom-bap loop)
]

PROGRESSIONS_TRAP = [
    [1, 6, 7, 6],       # i - VI - VII - VI (Metro Boomin dark anthem)
    [1, 2, 1, 7],       # i - II - i - VII (Phrygian tension vamp)
    [1, 6],             # i - VI (Hypnotic dark trap loop)
    [1, 7],             # i - VII (Heavy 808 drone loop)
    [1, 4, 6, 5],       # i - iv - VI - V (Dark cinematic turnaround)
]

PROGRESSIONS_MINIMAL = [
    [1, 4],             # i - iv (Four Tet deep modal vamp)
    [1, 5],             # i - v (Hypnotic UK Garage pulse)
    [1, 7, 4, 5],       # i - VII - iv - v (Dub techno / minimal house groove)
    [1, 6, 4, 7],       # i - VI - iv - VII (Atmospheric deep house)
]

PROGRESSIONS_AMBIENT = [
    [1, 5, 4, 1],       # I - V - IV - I (Brian Eno suspended calm)
    [1, 7, 6, 7],       # i - VII - VI - VII (Cinematic Hans Zimmer drone arc)
    [1, 4, 1, 5],       # Suspended non-resolving harmonic cloud
    [1, 6],             # Vast two-chord ocean
]


@dataclass
class Chord:
    root_midi: int              # Bass root note (e.g. 48 for C3)
    midi_notes: List[int]       # Chord voicings (e.g. [48, 55, 60, 64] for C3, G3, C4, E4)
    frequencies: List[float]    # In Hz
    degree: int                 # 1 to 7


def generate_chord_voicing(scale: MusicalScale, degree: int, base_octave: int = 3) -> Chord:
    """
    Generate an acoustically balanced, wide open-voicing triad/seventh chord
    to prevent muddy mid-frequency clutter.
    """
    # Root note in bass octave (e.g. octave 2 or 3)
    root_deg = degree
    third_deg = degree + 2
    fifth_deg = degree + 4
    seventh_deg = degree + 6

    def deg_to_midi(deg: int, oct_shift: int) -> int:
        idx = (deg - 1) % len(scale.intervals)
        oct_wrap = (deg - 1) // len(scale.intervals)
        root_pc = scale.pitch_classes[0]
        pc = (root_pc + scale.intervals[idx]) % 12
        return (base_octave + oct_shift + 1) * 12 + pc

    bass_note = deg_to_midi(root_deg, -1)   # Low bass note
    fifth_note = deg_to_midi(fifth_deg, 0)  # Mid-low fifth
    root_mid = deg_to_midi(root_deg, 0)     # Mid octave root
    third_high = deg_to_midi(third_deg, 1)  # High third

    chord_midis = [bass_note, fifth_note, root_mid, third_high]
    freqs = [midi_to_hz(m) for m in chord_midis]

    return Chord(
        root_midi=bass_note,
        midi_notes=chord_midis,
        frequencies=freqs,
        degree=degree
    )


def select_chord_progression(
    scale: MusicalScale,
    rng: Optional[SeededRNG] = None,
    genre_preference: Optional[str] = None
) -> List[Chord]:
    """Select and voice a procedural chord progression matched to the scale and genre."""
    if rng is None:
        rng = SeededRNG()

    pref = (genre_preference or "").lower()
    is_minor = "minor" in scale.scale_type or scale.scale_type in ["dorian", "phrygian", "blues"]

    if pref in ["trap", "drill"]:
        degrees = rng.choice(PROGRESSIONS_TRAP)
    elif pref in ["pop", "dance", "synthpop"]:
        degrees = rng.choice(PROGRESSIONS_POP)
    elif pref in ["hiphop", "hip_hop", "boom_bap", "lofi"]:
        degrees = rng.choice(PROGRESSIONS_HIPHOP)
    elif pref in ["rap", "electro", "french_touch"]:
        degrees = rng.choice(PROGRESSIONS_RAP)
    elif pref in ["minimal", "house", "garage"]:
        degrees = rng.choice(PROGRESSIONS_MINIMAL)
    elif pref in ["none", "ambient"]:
        degrees = rng.choice(PROGRESSIONS_AMBIENT)
    elif is_minor:
        degrees = rng.choice(PROGRESSIONS_MINOR)
    else:
        degrees = rng.choice(PROGRESSIONS_MAJOR)

    chords = [generate_chord_voicing(scale, deg) for deg in degrees]
    return chords
