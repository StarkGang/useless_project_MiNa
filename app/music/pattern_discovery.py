"""
Pattern Discovery & Organic Repetition Engine
TheUnnecessaryFM Philosophy:
  1. Find repetition that already exists in the recording.
  2. Create sparse, breathing repetition when natural repetition doesn't exist.
  3. Never use generic robotic drum grids or synthetic drum machines.
  4. Controlled variation (A, A', A, A'', A', A) and natural humanized timing.
  5. Deliberate use of silence.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from ..dsp.segmentation import AudioSlice, SourcePalette
from ..utils.random import SeededRNG


@dataclass
class PatternEvent:
    time_offset: float              # Offset in seconds within the bar/phrase
    slice_obj: AudioSlice           # The chosen source slice
    gain: float                     # Relative gain [0.0, 1.0]
    pan: float                      # Stereo position [-1.0, 1.0]
    subtle_pitch_ratio: float       # Subtle pitch micro-variation (0.96 to 1.04)
    lowpass_hz: Optional[float]     # Subtle filter variation if any


@dataclass
class EvolvingBar:
    bar_index: int
    duration: float                 # Duration of this bar in seconds
    events: List[PatternEvent]


@dataclass
class OrganicGroove:
    is_naturally_rhythmic: bool
    tempo_bpm: float
    seconds_per_bar: float
    anchor_slice: AudioSlice
    counter_slice: AudioSlice
    accent_slice: AudioSlice
    evolving_bars: List[EvolvingBar] # 4-8 evolving bars with natural variations


def discover_or_create_groove(
    palette: SourcePalette,
    bpm: float,
    has_natural_rhythm: bool = False,
    rng: Optional[SeededRNG] = None
) -> OrganicGroove:
    """
    Analyzes existing source characteristics or builds an organic, evolving
    percussive phrase strictly using the source audio slices.
    """
    if rng is None:
        rng = SeededRNG(42)

    seconds_per_bar = (60.0 / bpm) * 4.0
    step_duration = seconds_per_bar / 16.0

    # Pick 2-4 primary contrasting source elements
    # 1. Anchor: low/mid transient or solid impact
    anchor = rng.choice(palette.impacts)
    # 2. Counter: crisper click, tick, or high-energy pulse
    counter = rng.choice(palette.pulses)
    # 3. Accent: striking transient or dynamic movement
    accent = rng.choice(palette.accents)
    # 4. Ghost / Texture: subtle micro-event
    ghost = rng.choice(palette.textures) if palette.textures else counter

    # Base 16-step rhythmic skeleton (sparse, breathing, syncopated)
    # 1 = Anchor, 2 = Counter, 3 = Accent, 4 = Ghost, 0 = Silence
    skeletons = [
        # Sparse syncopated groove
        [1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 1, 0, 4, 0],
        # Breathing minimalist pulse
        [1, 0, 0, 4, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0],
        # Downbeat heavy with late syncopation
        [1, 0, 4, 0, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 3, 0],
        # Subtle ambient pulse
        [1, 0, 0, 0, 0, 4, 2, 0, 0, 0, 1, 0, 2, 0, 0, 0],
    ]
    base_pattern = rng.choice(skeletons)

    num_bars = 4
    evolving_bars: List[EvolvingBar] = []

    for b in range(num_bars):
        bar_events: List[PatternEvent] = []

        # Create subtle structural variation across bars (A, A', B, A'')
        bar_variation = base_pattern.copy()
        if b == 1:
            # Bar 2: add a subtle ghost note or drop step 12
            if rng.chance(0.6):
                bar_variation[10] = 4
            if rng.chance(0.4):
                bar_variation[14] = 2
        elif b == 2:
            # Bar 3: introduce accent or shift syncopation
            bar_variation[12] = 3 if rng.chance(0.7) else 1
            if rng.chance(0.5):
                bar_variation[6] = 2
        elif b == 3:
            # Bar 4: breathe - drop a hit to create anticipation, or add a quick pickup
            if rng.chance(0.5):
                bar_variation[4] = 0 # Breathing silence
            if rng.chance(0.7):
                bar_variation[15] = 4 # Pickup ghost note

        for step_idx, code in enumerate(bar_variation):
            if code == 0:
                continue

            # Target nominal time
            nominal_time = step_idx * step_duration

            # Humanized micro-timing jitter (-12ms to +14ms)
            # NEVER rigidly quantized
            jitter_sec = rng.uniform(-0.012, 0.014)
            actual_time = max(0.0, nominal_time + jitter_sec)

            # Map code to slice and expressive micro-variations
            if code == 1:
                sl = anchor
                gain = rng.uniform(0.75, 0.95)
                pan = rng.uniform(-0.15, 0.15)
                pitch = rng.uniform(0.98, 1.02)
            elif code == 2:
                sl = counter
                gain = rng.uniform(0.55, 0.82)
                pan = rng.uniform(-0.45, 0.45)
                pitch = rng.uniform(0.97, 1.03)
            elif code == 3:
                sl = accent
                gain = rng.uniform(0.80, 1.0)
                pan = rng.uniform(-0.25, 0.25)
                pitch = rng.uniform(0.95, 1.02)
            else: # Ghost note
                sl = ghost
                gain = rng.uniform(0.30, 0.55)
                pan = rng.uniform(-0.55, 0.55)
                pitch = rng.uniform(0.96, 1.04)

            # Subtle filter variation (e.g. slight warm lowpass on ghost hits)
            lp_hz = rng.uniform(3500.0, 8000.0) if code == 4 else None

            bar_events.append(PatternEvent(
                time_offset=actual_time,
                slice_obj=sl,
                gain=round(gain, 3),
                pan=round(pan, 3),
                subtle_pitch_ratio=round(pitch, 4),
                lowpass_hz=lp_hz
            ))

        evolving_bars.append(EvolvingBar(
            bar_index=b,
            duration=seconds_per_bar,
            events=bar_events
        ))

    return OrganicGroove(
        is_naturally_rhythmic=has_natural_rhythm,
        tempo_bpm=bpm,
        seconds_per_bar=seconds_per_bar,
        anchor_slice=anchor,
        counter_slice=counter,
        accent_slice=accent,
        evolving_bars=evolving_bars
    )
