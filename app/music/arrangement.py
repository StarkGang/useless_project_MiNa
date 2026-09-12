"""
Procedural Musical Arrangement and Adaptive Form Engine
TheUnnecessaryFM Philosophy:
  - Build music through authentic arrangement, dynamics, and silence, modeled after legendary reference songs.
  - MixGraph & Sound On Sound track-by-track structure modeling.
  - Generates constrained structures targeting 30s or 60s.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from ..utils.random import SeededRNG


@dataclass
class Section:
    name: str                         # E.g. "Stutter_Hook_Intro", "Drums_Solo_Intro", "Chorus_Supersaw_Climax"
    start_time: float                 # Seconds
    duration: float                   # Seconds
    bars: int                         # Number of bars
    energy_level: float               # Target intensity [0.0, 1.0]
    active_layers: List[str]          # "bed", "drums", "bass", "chords", "melody", "accents", "riser"
    filter_cutoff: float              # Master/bus filter brightness in Hz


@dataclass
class CompositionArrangement:
    tempo_bpm: float
    total_bars: int
    seconds_per_bar: float
    total_duration: float             # In seconds (target 30s or 60s)
    sections: List[Section]
    selected_form: str
    available_instruments: List[str]


# Authentic Reference Song Flow Templates (MixGraph & Sound On Sound modeled)
FLOW_TEMPLATES = {
    # 1. Lady Gaga & RedOne ("Poker Face" / "Just Dance"):
    # Source tag anchor + stutter saw intro -> stripped 16th verse with vocal hook -> white noise riser -> explosive chorus
    "hook_first_explosion": {
        "sections": [
            ("Stutter_Hook_Intro", 0.16, ["source_tag", "bed", "melody", "bass"], 6500.0, 0.65),
            ("Verse_Electro_Drop", 0.28, ["drums", "bass", "chords", "vocal_hook"], 10000.0, 0.82),
            ("Pre_Chorus_Build", 0.14, ["drums", "chords", "riser"], 13500.0, 0.88),
            ("Chorus_Supersaw_Climax", 0.30, ["drums", "bass", "chords", "melody", "vocal_hook", "accents"], 18500.0, 1.00),
            ("Outro_Hook", 0.12, ["drums", "melody", "bass", "vocal_hook"], 8000.0, 0.70)
        ]
    },

    # 2. Michael Jackson & Quincy Jones ("Billie Jean" / "Thriller"):
    # Source tag intro -> pure drums -> walking ostinato bass entry -> rhythm chops -> vocal hook groove
    "drums_first_build": {
        "sections": [
            ("Drums_Solo_Intro", 0.14, ["source_tag", "drums"], 8000.0, 0.60),
            ("Ostinato_Bass_Entry", 0.16, ["drums", "bass", "vocal_hook"], 9500.0, 0.75),
            ("Verse_Pocket", 0.26, ["drums", "bass", "chords", "vocal_hook"], 11000.0, 0.84),
            ("Pre_Chorus_Swell", 0.14, ["drums", "bass", "chords", "riser"], 13000.0, 0.88),
            ("Chorus_Climax", 0.20, ["drums", "bass", "chords", "melody", "vocal_hook", "accents"], 17000.0, 0.98),
            ("Outro_Groove", 0.10, ["drums", "bass", "vocal_hook"], 8500.0, 0.65)
        ]
    },

    # 3. Kanye West & Mike Dean ("Flashing Lights"):
    # Source tag anchor + staccato strings solo -> heavy 808 drop with soulful vocal chops -> full chorus
    "staccato_strings_drop": {
        "sections": [
            ("Iconic_Strings_Intro", 0.16, ["source_tag", "bed", "melody", "chords"], 7000.0, 0.65),
            ("Heavy_808_Verse_Drop", 0.28, ["drums", "bass", "chords", "vocal_hook"], 11000.0, 0.86),
            ("Pre_Chorus_Tension", 0.14, ["drums", "melody", "chords", "riser"], 13500.0, 0.88),
            ("Chorus_Lush_Hook", 0.30, ["drums", "bass", "chords", "melody", "vocal_hook", "accents"], 17500.0, 1.00),
            ("Outro_Strings_Decay", 0.12, ["chords", "melody", "vocal_hook"], 6000.0, 0.55)
        ]
    },

    # 4. Bruno Mars & Mark Ronson ("Uptown Funk" / "24K Magic"):
    # Source tag intro -> slap bass + swung live funk kit -> vocal chant hooks -> brass drop
    "funk_vamp_drop": {
        "sections": [
            ("Funky_Chant_Intro", 0.15, ["source_tag", "bed", "drums"], 7500.0, 0.60),
            ("Slap_Bass_Verse_Vamp", 0.28, ["drums", "bass", "chords", "vocal_hook"], 11500.0, 0.84),
            ("Stop_Time_Break", 0.15, ["drums", "chords", "riser"], 14000.0, 0.88),
            ("Chorus_Brass_Drop", 0.30, ["drums", "bass", "chords", "melody", "vocal_hook", "accents"], 18500.0, 1.00),
            ("Outro_Vamp", 0.12, ["drums", "bass", "melody", "vocal_hook"], 9000.0, 0.72)
        ]
    },

    # 5. The Weeknd & Max Martin ("Blinding Lights"):
    # Source tag pulse -> Juno lead drop -> driving bass verse with vocal chop accents -> full anthem
    "synthwave_driving_drop": {
        "sections": [
            ("Filtered_Pulse_Intro", 0.14, ["source_tag", "bed", "bass"], 6000.0, 0.60),
            ("Intro_Juno_Hook", 0.16, ["drums", "bass", "melody"], 11000.0, 0.86),
            ("Verse_Driving_Pulse", 0.26, ["drums", "bass", "chords", "vocal_hook"], 10500.0, 0.82),
            ("Pre_Chorus_Snare_Roll", 0.14, ["drums", "chords", "riser"], 14000.0, 0.90),
            ("Chorus_Synth_Climax", 0.20, ["drums", "bass", "chords", "melody", "vocal_hook", "accents"], 19000.0, 1.00),
            ("Outro_Reverb_Tail", 0.10, ["melody", "chords"], 7000.0, 0.55)
        ]
    },

    # 6. Dr. Dre & Scott Storch ("Still D.R.E."):
    # Source tag anchor + staccato piano -> boom bap sub drop with vocal chops -> soaring G-funk whistle
    "piano_whistle_drop": {
        "sections": [
            ("Staccato_Piano_Intro", 0.16, ["source_tag", "chords"], 8000.0, 0.62),
            ("Boom_Bap_Sub_Drop", 0.28, ["drums", "bass", "chords", "vocal_hook"], 11000.0, 0.84),
            ("G_Funk_Whistle_Flight", 0.32, ["drums", "bass", "chords", "melody", "vocal_hook", "accents"], 16500.0, 0.96),
            ("Outro_Piano_Fade", 0.24, ["chords", "bass"], 7500.0, 0.60)
        ]
    },

    # 7. Dua Lipa ("Don't Start Now" / "Levitating"):
    # Source tag teaser -> 4-on-floor club drop -> percussive vocal chops -> full disco explosion
    "disco_pump_build": {
        "sections": [
            ("Disco_Bass_Tease", 0.15, ["source_tag", "bed", "bass"], 6500.0, 0.62),
            ("Club_Boots_Cats_Drop", 0.28, ["drums", "bass", "chords", "vocal_hook"], 12000.0, 0.85),
            ("Upbeat_Open_Hat_Build", 0.15, ["drums", "chords", "riser"], 14000.0, 0.90),
            ("Nu_Disco_Full_Climax", 0.30, ["drums", "bass", "chords", "melody", "vocal_hook", "accents"], 18500.0, 1.00),
            ("Outro_Octave_Pump", 0.12, ["drums", "bass", "vocal_hook"], 8500.0, 0.68)
        ]
    },

    # 8. Metro Boomin ("Dark Cinematic Trap"):
    # Source tag piano atmosphere -> sliding 808 drop with eerie vocal chops -> trap climax
    "dark_cinematic_drop": {
        "sections": [
            ("Minor_Piano_Atmosphere", 0.16, ["source_tag", "bed", "chords"], 6000.0, 0.60),
            ("Sliding_808_Drop", 0.28, ["drums", "bass", "chords", "vocal_hook"], 10500.0, 0.86),
            ("Pitched_Snare_Roll_Build", 0.14, ["drums", "chords", "riser"], 14000.0, 0.90),
            ("Trap_Chorus_Climax", 0.30, ["drums", "bass", "chords", "melody", "vocal_hook", "accents"], 17500.0, 0.98),
            ("Outro_Sub_Fade", 0.12, ["chords", "bass", "vocal_hook"], 7000.0, 0.55)
        ]
    },

    # 9. Standard / Ambient:
    "standard_cinematic": {
        "sections": [
            ("Intro_Reveal", 0.15, ["source_tag", "bed", "chords"], 6000.0, 0.55),
            ("Groove_Drop", 0.28, ["bed", "drums", "bass", "chords", "vocal_hook"], 10000.0, 0.80),
            ("Build_Up", 0.15, ["bed", "drums", "chords", "riser"], 13000.0, 0.86),
            ("Chorus_Climax", 0.30, ["bed", "drums", "bass", "chords", "melody", "vocal_hook", "accents"], 18000.0, 0.96),
            ("Outro_Resolution", 0.12, ["bed", "chords", "bass", "vocal_hook", "accents"], 8000.0, 0.60)
        ]
    }
}


def plan_arrangement(
    bpm: float,
    source_energy_envelope: Optional[List[float]] = None,
    sonic_character: str = "MIXED",
    rng: Optional[SeededRNG] = None,
    target_sec: float = 30.0,
    flow_archetype: str = "standard_cinematic"
) -> CompositionArrangement:
    """
    Plans a musical composition targeting the requested duration (30s or 60s)
    using authentic reference song flow graphs (MixGraph & Sound On Sound modeled).
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
    low_bound = target_sec * 0.80
    high_bound = target_sec * 1.20
    if total_duration < low_bound:
        ideal_bars += 2
        total_duration = ideal_bars * seconds_per_bar
    elif total_duration > high_bound and ideal_bars > min_bars:
        ideal_bars -= 2
        total_duration = ideal_bars * seconds_per_bar

    # Select Flow Template based on artist archetype
    template = FLOW_TEMPLATES.get(flow_archetype, FLOW_TEMPLATES["standard_cinematic"])
    sec_defs = template["sections"]
    num_sections = len(sec_defs)

    bar_distribution = [d[1] for d in sec_defs]
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

    for i, (name, _, layers, base_cutoff, base_energy) in enumerate(sec_defs):
        bars = raw_bars[i]
        sec_dur = bars * seconds_per_bar

        progress_idx = int(round((current_time / total_duration) * (len(env) - 1)))
        progress_idx = min(len(env) - 1, max(0, progress_idx))
        source_intensity = env[progress_idx]

        # Blend base template energy with input audio dynamics
        energy = np.clip(base_energy * (0.80 + 0.35 * source_intensity), 0.35, 1.0)
        cutoff = float(np.clip(base_cutoff * (0.85 + 0.25 * source_intensity), 3000.0, 20000.0))

        sections.append(Section(
            name=name,
            start_time=round(current_time, 3),
            duration=round(sec_dur, 3),
            bars=bars,
            energy_level=round(float(energy), 2),
            active_layers=list(layers),
            filter_cutoff=round(cutoff, 1)
        ))
        current_time += sec_dur

    return CompositionArrangement(
        tempo_bpm=bpm,
        total_bars=ideal_bars,
        seconds_per_bar=round(seconds_per_bar, 4),
        total_duration=round(total_duration, 2),
        sections=sections,
        selected_form=flow_archetype,
        available_instruments=["drums", "bass", "chords", "melody", "bed", "accents", "riser"]
    )
