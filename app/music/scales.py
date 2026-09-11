"""
Procedural Musical Scales and Tonality Engine
Defines scales, modes, note-to-frequency conversions, and intelligent weighted scale selection
based on source spectral brightness and noisiness.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..utils.random import SeededRNG

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale semitone intervals from root
SCALE_DEFINITIONS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10]
}


@dataclass
class MusicalScale:
    name: str                         # E.g. "D Minor"
    scale_type: str                   # E.g. "minor"
    root_name: str                    # E.g. "D"
    root_midi: int                    # E.g. 62 (D4)
    intervals: List[int]              # Semitone offsets from root
    pitch_classes: List[int]          # 0..11 pitch classes

    def get_notes_in_octave_range(self, min_midi: int = 36, max_midi: int = 84) -> List[int]:
        """Return all MIDI note numbers in the scale within the specified range."""
        notes = []
        for midi in range(min_midi, max_midi + 1):
            pc = midi % 12
            if pc in self.pitch_classes:
                notes.append(midi)
        return notes

    def get_scale_degree(self, degree: int, octave: int = 4) -> int:
        """degree: 1 to N (1 = root, 3 = third, 5 = fifth, etc.)."""
        idx = (degree - 1) % len(self.intervals)
        octave_offset = (degree - 1) // len(self.intervals)
        return (octave + 1) * 12 + (self.pitch_classes[0] + self.intervals[idx]) % 12 + octave_offset * 12


def select_scale(
    brightness: float,
    noisiness: float,
    candidate_root_notes: Optional[List[str]] = None,
    rng: Optional[SeededRNG] = None,
    genre_preference: Optional[str] = None
) -> MusicalScale:
    """
    Intelligent weighted scale selection (Sections 11 & 12).
    - Source is dark (low centroid/brightness) -> favor minor, dorian, phrygian
    - Source is bright -> favor major, pentatonic_major, mixolydian
    - Source is extremely noisy -> favor pentatonic_minor, ambient dorian, blues
    - Genre preference:
      * hiphop: favor dorian, minor, pentatonic_minor, blues
      * rap: favor harmonic_minor, phrygian, minor, pentatonic_minor
      * pop: favor major, pentatonic_major, mixolydian, minor
    """
    if rng is None:
        rng = SeededRNG()

    scale_types = list(SCALE_DEFINITIONS.keys())
    pref = (genre_preference or "").lower()

    # Build weights based on source features and genre directives
    weights = []
    for st in scale_types:
        w = 1.0

        if pref in ["hiphop", "hip_hop", "boom_bap", "lofi"]:
            if st in ["dorian", "pentatonic_minor"]:
                w += 4.0
            elif st in ["minor", "blues"]:
                w += 3.0
            elif st in ["mixolydian"]:
                w += 1.5
        elif pref in ["rap", "trap", "drill"]:
            if st in ["harmonic_minor", "phrygian"]:
                w += 5.0
            elif st in ["minor", "pentatonic_minor"]:
                w += 3.5
            elif st in ["blues"]:
                w += 1.5
        elif pref in ["pop", "dance", "synthpop"]:
            if st in ["major", "pentatonic_major"]:
                w += 4.5
            elif st in ["mixolydian"]:
                w += 3.5
            elif st in ["minor"]:
                w += 2.0
        else:
            if brightness > 0.55:
                if st in ["major", "pentatonic_major", "lydian"]:
                    w += 2.5
                elif st in ["mixolydian"]:
                    w += 1.5
            else:
                if st in ["minor", "dorian", "harmonic_minor"]:
                    w += 2.5
                elif st in ["phrygian", "pentatonic_minor"]:
                    w += 1.5

            if noisiness > 0.4:
                if st in ["pentatonic_minor", "dorian", "blues"]:
                    w += 2.0

        weights.append(w)

    chosen_type = rng.choice(scale_types, weights=weights)

    # Root note selection: prioritize candidate roots if detected, else weighted random
    ENHARMONICS = {"Bb": "A#", "Eb": "D#", "Ab": "G#", "Db": "C#", "Gb": "F#"}
    if candidate_root_notes and len(candidate_root_notes) > 0 and candidate_root_notes[0] != "None":
        root_name = candidate_root_notes[0]
        root_name = ENHARMONICS.get(root_name, root_name)
        if root_name in NOTE_NAMES:
            root_idx = NOTE_NAMES.index(root_name)
        else:
            root_name = "C"
            root_idx = 0
    else:
        # Common pleasant musical roots matched to genre
        if pref in ["rap", "trap", "drill"]:
            popular_roots = ["C#", "D", "D#", "F", "F#", "G#", "A"]
        elif pref in ["hiphop", "hip_hop", "lofi"]:
            popular_roots = ["C", "D", "E", "F", "G", "A", "A#"]
        elif pref in ["pop"]:
            popular_roots = ["C", "D", "E", "F", "G", "A"]
        else:
            popular_roots = ["C", "D", "E", "F", "G", "A", "A#"]
        root_name = rng.choice(popular_roots)
        root_idx = NOTE_NAMES.index(root_name)

    intervals = SCALE_DEFINITIONS[chosen_type]
    pitch_classes = [(root_idx + interval) % 12 for interval in intervals]
    root_midi = 60 + root_idx

    display_name = f"{root_name} {chosen_type.replace('_', ' ').title()}"

    return MusicalScale(
        name=display_name,
        scale_type=chosen_type,
        root_name=root_name,
        root_midi=root_midi,
        intervals=intervals,
        pitch_classes=pitch_classes
    )
