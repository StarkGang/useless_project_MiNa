"""
Procedural Sound Synthesis Engine for TheUnnecessaryFM
Pure mathematical synthesis without external audio files or neural models.
Provides:
  - Oscillators: Sine, Saw, Square, Triangle, Noise, FM Synthesis
  - ADSR Envelope Generator (linear & exponential)
  - Virtual Instruments: SineBass, SoftBass, Pluck, Bell, Pad, Drone,
    NoisePercussion, FMLead, SoftLead, StringLike
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy import signal


def adsr_envelope(
    duration: float,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    sr: int = 44100
) -> np.ndarray:
    """Generate an ADSR amplitude envelope array."""
    total_samples = int(duration * sr)
    if total_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    n_attack = max(1, int(attack * sr))
    n_decay = max(1, int(decay * sr))
    n_release = max(1, int(release * sr))

    # Sustain duration fills remainder
    n_sustain = max(0, total_samples - n_attack - n_decay - n_release)

    # If note is shorter than attack + decay + release, scale proportionally
    if n_attack + n_decay + n_release > total_samples:
        scale = total_samples / (n_attack + n_decay + n_release)
        n_attack = max(1, int(n_attack * scale))
        n_decay = max(1, int(n_decay * scale))
        n_release = max(1, total_samples - n_attack - n_decay)
        n_sustain = 0

    attack_curve = np.linspace(0.0, 1.0, n_attack)
    decay_curve = np.linspace(1.0, sustain, n_decay)
    sustain_curve = np.full(n_sustain, sustain)
    release_curve = np.linspace(sustain, 0.0, n_release)

    env = np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])
    if len(env) < total_samples:
        env = np.pad(env, (0, total_samples - len(env)))
    elif len(env) > total_samples:
        env = env[:total_samples]

    return env.astype(np.float32)


def sine_wave(freq: float, duration: float, phase: float = 0.0, sr: int = 44100) -> np.ndarray:
    """Generate pure sine wave oscillator."""
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    return np.sin(2.0 * np.pi * freq * t + phase).astype(np.float32)


def saw_wave(freq: float, duration: float, sr: int = 44100) -> np.ndarray:
    """Generate anti-aliased saw wave using band-limited additive synthesis (12 harmonics)."""
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    out = np.zeros(num_samples, dtype=np.float32)
    max_h = min(20, int((sr * 0.45) / max(1.0, freq)))
    for h in range(1, max_h + 1):
        out += (1.0 / h) * np.sin(2.0 * np.pi * freq * h * t)
    pk = np.max(np.abs(out))
    if pk > 0:
        out = (out / pk) * 0.95
    return out.astype(np.float32)


def square_wave(freq: float, duration: float, sr: int = 44100) -> np.ndarray:
    """Generate band-limited square wave using odd harmonics."""
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    out = np.zeros(num_samples, dtype=np.float32)
    max_h = min(15, int((sr * 0.45) / max(1.0, freq)))
    for h in range(1, max_h + 1, 2):
        out += (1.0 / h) * np.sin(2.0 * np.pi * freq * h * t)
    pk = np.max(np.abs(out))
    if pk > 0:
        out = (out / pk) * 0.95
    return out.astype(np.float32)


def triangle_wave(freq: float, duration: float, sr: int = 44100) -> np.ndarray:
    """Generate band-limited triangle wave."""
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    out = np.zeros(num_samples, dtype=np.float32)
    max_h = min(15, int((sr * 0.45) / max(1.0, freq)))
    sign = 1.0
    for h in range(1, max_h + 1, 2):
        out += sign * (1.0 / (h ** 2)) * np.sin(2.0 * np.pi * freq * h * t)
        sign *= -1.0
    pk = np.max(np.abs(out))
    if pk > 0:
        out = (out / pk) * 0.95
    return out.astype(np.float32)


def fm_synth(
    carrier_freq: float,
    mod_freq: float,
    mod_index: float,
    duration: float,
    sr: int = 44100
) -> np.ndarray:
    """Frequency Modulation synthesis oscillator."""
    num_samples = int(duration * sr)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    # Modulator
    modulator = mod_index * np.sin(2.0 * np.pi * mod_freq * t)
    # Carrier
    carrier = np.sin(2.0 * np.pi * carrier_freq * t + modulator)
    return carrier.astype(np.float32)


# ==============================================================================
# Procedural Virtual Instruments
# ==============================================================================

class SynthInstrument:
    """Base class for procedurally synthesized instruments."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        raise NotImplementedError


class SineBass(SynthInstrument):
    """Deep pure fundamental sub-bass with subtle 2nd harmonic saturation."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        # Keep bass frequency in musical sub register (35 Hz - 180 Hz)
        while freq > 180.0:
            freq /= 2.0
        while freq < 35.0:
            freq *= 2.0

        osc = sine_wave(freq, duration, sr=sr)
        # Add slight 2nd harmonic warmth
        osc += 0.15 * sine_wave(freq * 2.0, duration, sr=sr)
        # Soft saturation
        osc = np.tanh(osc * 1.4)

        env = adsr_envelope(duration, attack=0.015, decay=0.2, sustain=0.75, release=0.08, sr=sr)
        return (osc * env * velocity).astype(np.float32)


class SoftBass(SynthInstrument):
    """Warm filtered saw/triangle bass."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        while freq > 200.0:
            freq /= 2.0
        while freq < 40.0:
            freq *= 2.0

        saw = saw_wave(freq, duration, sr=sr)
        sub = sine_wave(freq, duration, sr=sr)
        raw = saw * 0.4 + sub * 0.6

        # Low-pass filter at 350 Hz
        b, a = signal.butter(2, min(0.45, 380.0 / (sr * 0.5)), btype='low')
        filtered = signal.lfilter(b, a, raw)

        env = adsr_envelope(duration, attack=0.01, decay=0.15, sustain=0.7, release=0.1, sr=sr)
        return (filtered * env * velocity).astype(np.float32)


class Pluck(SynthInstrument):
    """Karplus-Strong physical modeling pluck via fast vectorized feedback filter."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        total_samples = int(duration * sr)
        if total_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        period = int(round(sr / max(20.0, freq)))
        if period <= 0:
            period = 100

        # Excitation: short burst of white noise of length 'period'
        burst = np.random.uniform(-1.0, 1.0, period).astype(np.float32)
        excitation = np.pad(burst, (0, max(0, total_samples - period)))[:total_samples]

        # Feedback loop: y[n] = x[n] + decay * y[n - period]
        decay_factor = float(np.clip(0.988 - (freq / 8000.0) * 0.05, 0.90, 0.995))
        b = np.array([1.0], dtype=np.float32)
        a = np.zeros(period + 1, dtype=np.float32)
        a[0] = 1.0
        a[period] = -decay_factor

        out = signal.lfilter(b, a, excitation)
        # Gentle low-pass filter for acoustic warmth
        cutoff = min(sr * 0.45, max(800.0, freq * 4.5))
        b_lp, a_lp = signal.butter(1, cutoff / (sr * 0.5), btype='low')
        out = signal.lfilter(b_lp, a_lp, out)

        fade_samples = min(256, total_samples)
        out[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

        pk = np.max(np.abs(out))
        if pk > 0:
            out = (out / pk) * velocity * 0.85
        return out.astype(np.float32)


class Bell(SynthInstrument):
    """FM synthesized metallic chime/bell."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        # Non-integer modulator ratio (2.76) for metallic bell timbre
        mod_ratio = 2.756
        carrier = fm_synth(
            carrier_freq=freq,
            mod_freq=freq * mod_ratio,
            mod_index=2.5,
            duration=duration,
            sr=sr
        )
        # Fast attack, exponential decay
        env = adsr_envelope(duration, attack=0.003, decay=duration * 0.8, sustain=0.05, release=0.1, sr=sr)
        return (carrier * env * velocity * 0.7).astype(np.float32)


class Pad(SynthInstrument):
    """Rich polyphonic warm pad with slight chorus detune."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        osc1 = triangle_wave(freq, duration, sr=sr)
        osc2 = sine_wave(freq * 1.003, duration, sr=sr)   # Slight detune
        osc3 = saw_wave(freq * 0.997, duration, sr=sr) * 0.3

        raw = osc1 * 0.45 + osc2 * 0.45 + osc3

        # Lowpass filter
        b, a = signal.butter(2, min(0.45, 1400.0 / (sr * 0.5)), btype='low')
        filtered = signal.lfilter(b, a, raw)

        env = adsr_envelope(duration, attack=0.45, decay=0.3, sustain=0.8, release=0.5, sr=sr)
        return (filtered * env * velocity * 0.7).astype(np.float32)


class Drone(SynthInstrument):
    """Deep modal drone with slow pulsating harmonics."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        while freq > 130.0:
            freq /= 2.0

        osc = sine_wave(freq, duration, sr=sr) * 0.6
        osc += sine_wave(freq * 1.5, duration, sr=sr) * 0.25  # Fifth
        osc += sine_wave(freq * 2.0, duration, sr=sr) * 0.15  # Octave

        # Slow LFO (0.2 Hz)
        t = np.linspace(0, duration, len(osc), endpoint=False)
        lfo = 0.75 + 0.25 * np.sin(2.0 * np.pi * 0.2 * t)
        osc = osc * lfo

        env = adsr_envelope(duration, attack=0.8, decay=0.2, sustain=0.9, release=0.8, sr=sr)
        return (osc * env * velocity * 0.75).astype(np.float32)


class FMLead(SynthInstrument):
    """Bright expressive FM lead synth."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        # 1:1 or 1:2 harmonic ratio
        lead = fm_synth(
            carrier_freq=freq,
            mod_freq=freq * 2.0,
            mod_index=1.8,
            duration=duration,
            sr=sr
        )
        env = adsr_envelope(duration, attack=0.03, decay=0.15, sustain=0.6, release=0.12, sr=sr)
        return (lead * env * velocity * 0.7).astype(np.float32)


class SoftLead(SynthInstrument):
    """Mellow flute-like triangle lead with gentle vibrato."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        # 5 Hz vibrato
        vibrato = 0.008 * np.sin(2.0 * np.pi * 5.0 * t)
        phase = np.cumsum(2.0 * np.pi * freq * (1.0 + vibrato) / sr)
        osc = np.sin(phase) + 0.25 * np.sin(phase * 2.0)

        env = adsr_envelope(duration, attack=0.08, decay=0.1, sustain=0.85, release=0.15, sr=sr)
        return (osc * env * velocity * 0.7).astype(np.float32)


class StringLike(SynthInstrument):
    """Bowed string emulation with warm saw cluster."""
    def render_note(self, freq: float, duration: float, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
        saw1 = saw_wave(freq, duration, sr=sr)
        saw2 = saw_wave(freq * 1.004, duration, sr=sr)
        raw = (saw1 + saw2) * 0.5

        # Resonant filter around 1200 Hz
        b, a = signal.butter(2, min(0.45, 1800.0 / (sr * 0.5)), btype='low')
        filtered = signal.lfilter(b, a, raw)

        env = adsr_envelope(duration, attack=0.15, decay=0.2, sustain=0.85, release=0.25, sr=sr)
        return (filtered * env * velocity * 0.65).astype(np.float32)


def get_procedural_instrument(name: str) -> SynthInstrument:
    """Factory helper to obtain instrument by name."""
    catalog = {
        "SineBass": SineBass,
        "SoftBass": SoftBass,
        "Pluck": Pluck,
        "Bell": Bell,
        "Pad": Pad,
        "Drone": Drone,
        "FMLead": FMLead,
        "SoftLead": SoftLead,
        "StringLike": StringLike,
    }
    cls = catalog.get(name, Pluck)
    return cls()


# ==============================================================================
# PROFESSIONAL NOISE-TO-MUSIC INSTRUMENT ENGINES
# (Directly converts raw audio/noise into tonal leads, plucks, bass, and drums)
# ==============================================================================

def render_noise_instrument_note(
    source_audio: np.ndarray,
    freq: float,
    duration: float,
    velocity: float = 0.85,
    style: str = "pluck",  # "pluck", "lead", "chime", "string"
    sr: int = 44100
) -> np.ndarray:
    """
    CRITICAL TRANSFORMATION:
    Converts a chunk of the source noise into a musical note with clear pitch
    while preserving the raw grit, acoustic texture, and identity of the noise.
    """
    total_samples = max(64, int(duration * sr))
    if len(source_audio) == 0:
        source_audio = np.random.normal(0, 0.1, total_samples).astype(np.float32)

    # 1. Extract a grain from the source noise
    if len(source_audio) < total_samples:
        reps = int(np.ceil(total_samples / len(source_audio)))
        noise_chunk = np.tile(source_audio, reps)[:total_samples]
    else:
        # Pick from a random position for natural variation
        start = np.random.randint(0, max(1, len(source_audio) - total_samples))
        noise_chunk = source_audio[start:start + total_samples].copy()

    # 2. Resonate at fundamental frequency via narrow 2nd-order bandpass filter
    freq = float(np.clip(freq, 40.0, sr * 0.45))
    q = 32.0 if style in ["pluck", "chime"] else 22.0
    bw = freq / q
    f_low = max(20.0, freq - bw * 0.5)
    f_high = min(sr * 0.48, freq + bw * 0.5)
    b, a = signal.butter(2, [f_low / (sr * 0.5), f_high / (sr * 0.5)], btype='bandpass')
    resonated_f0 = signal.lfilter(b, a, noise_chunk)

    # 3. Resonate at 2nd harmonic (octave) for acoustic warmth
    f2 = min(sr * 0.45, freq * 2.0)
    bw2 = f2 / 20.0
    b2, a2 = signal.butter(1, [max(20.0, f2 - bw2*0.5) / (sr * 0.5), min(sr*0.48, f2 + bw2*0.5) / (sr * 0.5)], btype='bandpass')
    resonated_f2 = signal.lfilter(b2, a2, noise_chunk)

    # Combine resonances (boost energy back)
    tonal_noise = (resonated_f0 * 18.0 + resonated_f2 * 6.0)

    # 4. Mix in raw noise texture (20%) so the source identity is distinctly heard!
    textured_note = tonal_noise * 0.80 + noise_chunk * 0.20

    # 5. Soft analog tape saturation
    textured_note = np.tanh(textured_note * 1.5)

    # 6. Apply musical envelope according to style
    if style == "pluck":
        env = adsr_envelope(duration, attack=0.004, decay=min(0.35, duration * 0.7), sustain=0.1, release=0.08, sr=sr)
    elif style == "chime":
        env = adsr_envelope(duration, attack=0.002, decay=duration * 0.8, sustain=0.05, release=0.15, sr=sr)
    elif style == "string":
        env = adsr_envelope(duration, attack=0.12, decay=0.2, sustain=0.85, release=0.2, sr=sr)
    else:  # "lead"
        env = adsr_envelope(duration, attack=0.03, decay=0.15, sustain=0.7, release=0.12, sr=sr)

    # Subtle sine undertone (15%) only for body
    sub_support = 0.15 * np.sin(2.0 * np.pi * freq * np.linspace(0, duration, total_samples, endpoint=False))

    rendered = (textured_note * env + sub_support * env) * velocity

    pk = np.max(np.abs(rendered))
    if pk > 0:
        rendered = (rendered / pk) * velocity * 0.9

    return rendered.astype(np.float32)


def render_noise_bass_note(
    source_audio: np.ndarray,
    freq: float,
    duration: float,
    velocity: float = 0.85,
    sr: int = 44100
) -> np.ndarray:
    """
    Hybrid Source Bass:
    Fuses the source recording's raw acoustic low-end texture with a warm,
    analog-saturated sub-bass oscillator at the musical root frequency.
    Carries the chord progression and anchors the groove.
    """
    while freq > 160.0:
        freq /= 2.0
    while freq < 38.0:
        freq *= 2.0

    total_samples = max(64, int(duration * sr))
    t = np.linspace(0, duration, total_samples, endpoint=False)

    # 1. Warm sub-bass fundamental (sine + subtle 2nd harmonic for small speakers)
    sub = np.sin(2.0 * np.pi * freq * t) * 0.75 + np.sin(4.0 * np.pi * freq * t) * 0.25

    # 2. Extract and saturate source audio low-end body
    if len(source_audio) < total_samples:
        reps = int(np.ceil(total_samples / max(1, len(source_audio))))
        noise_chunk = np.tile(source_audio, reps)[:total_samples]
    else:
        noise_chunk = source_audio[:total_samples].copy()

    b_lp, a_lp = signal.butter(2, min(0.45, 320.0 / (sr * 0.5)), btype='low')
    source_low = signal.lfilter(b_lp, a_lp, noise_chunk)
    p_src = np.max(np.abs(source_low))
    if p_src > 1e-4:
        source_low = (source_low / p_src)

    # 3. Fuse sub oscillator with source texture
    combined = sub * 0.65 + source_low * 0.45
    saturated = np.tanh(combined * 1.8)

    # 4. Punchy ADSR envelope with smooth release
    env = adsr_envelope(duration, attack=0.012, decay=0.15, sustain=0.82, release=0.08, sr=sr)
    out = (saturated * env * velocity).astype(np.float32)

    pk = np.max(np.abs(out))
    if pk > 0:
        out = (out / pk) * velocity * 0.90
    return out


def render_noise_kick(slice_audio: np.ndarray, velocity: float = 0.9, sr: int = 44100) -> np.ndarray:
    """
    Hybrid Source Kick:
    Combines the source transient attack click with an exponential pitch-sweep
    sub-bass body (140Hz -> 48Hz). Hits punchy in the chest while preserving
    the authentic acoustic signature of the uploaded noise.
    """
    dur = 0.32
    n_samples = int(dur * sr)
    t = np.linspace(0, dur, n_samples, endpoint=False)

    # 1. Source transient attack click (first 40ms)
    if len(slice_audio) < n_samples:
        slice_audio = np.pad(slice_audio, (0, n_samples - len(slice_audio)))
    else:
        slice_audio = slice_audio[:n_samples].copy()

    b_click, a_click = signal.butter(2, [80.0 / (sr * 0.5), min(sr * 0.48, 4000.0) / (sr * 0.5)], btype='bandpass')
    click_filtered = signal.lfilter(b_click, a_click, slice_audio)
    click_env = np.exp(-t * 45.0)
    source_click = click_filtered * click_env
    pk_c = np.max(np.abs(source_click))
    if pk_c > 1e-4:
        source_click = source_click / pk_c

    # 2. Punchy sub-bass pitch drop (140Hz -> 48Hz)
    f_start, f_end = 145.0, 46.0
    freq_curve = f_end + (f_start - f_end) * np.exp(-t * 26.0)
    phase = 2.0 * np.pi * np.cumsum(freq_curve) / sr
    sub_body = np.sin(phase) * np.exp(-t * 9.5)

    # 3. Layer and glue with soft-saturation
    kick = source_click * 0.40 + sub_body * 0.85
    kick = np.tanh(kick * 1.7)

    pk = np.max(np.abs(kick))
    if pk > 0:
        kick = (kick / pk) * velocity * 0.95
    return kick.astype(np.float32)


def render_noise_snare(slice_audio: np.ndarray, velocity: float = 0.8, sr: int = 44100) -> np.ndarray:
    """
    Hybrid Source Snare:
    Takes the source transient crack (bandpassed 350Hz-4.5kHz) and layers it with
    a resonant body tone (185Hz) and shaped snappy tail for a modern, crisp backbeat.
    """
    dur = 0.22
    n_samples = int(dur * sr)
    t = np.linspace(0, dur, n_samples, endpoint=False)

    if len(slice_audio) < n_samples:
        slice_audio = np.pad(slice_audio, (0, n_samples - len(slice_audio)))
    else:
        slice_audio = slice_audio[:n_samples].copy()

    # 1. Source acoustic crack
    b_sn, a_sn = signal.butter(2, [350.0 / (sr * 0.5), min(sr * 0.48, 5000.0) / (sr * 0.5)], btype='bandpass')
    source_crack = signal.lfilter(b_sn, a_sn, slice_audio)
    crack_env = np.exp(-t * 18.0)
    source_crack = source_crack * crack_env
    pk_cr = np.max(np.abs(source_crack))
    if pk_cr > 1e-4:
        source_crack = source_crack / pk_cr

    # 2. Resonant shell body tone (185Hz with fast decay)
    body_tone = np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-t * 32.0)

    # 3. Layer crack with shell body
    snare = source_crack * 0.65 + body_tone * 0.40
    snare = np.tanh(snare * 2.0)

    pk = np.max(np.abs(snare))
    if pk > 0:
        snare = (snare / pk) * velocity * 0.88
    return snare.astype(np.float32)


def render_noise_hihat(slice_audio: np.ndarray, velocity: float = 0.6, sr: int = 44100) -> np.ndarray:
    """
    Hybrid Source Hi-Hat / Shaker:
    High-passes micro-transients from the source audio above 5kHz with a crisp,
    snappy decay envelope for sizzling, grooving percussive rhythm.
    """
    dur = 0.08
    n_samples = int(dur * sr)
    t = np.linspace(0, dur, n_samples, endpoint=False)

    if len(slice_audio) < n_samples:
        slice_audio = np.pad(slice_audio, (0, n_samples - len(slice_audio)))
    else:
        slice_audio = slice_audio[:n_samples].copy()

    # Highpass source noise
    b_hp, a_hp = signal.butter(2, min(0.45, 5200.0 / (sr * 0.5)), btype='high')
    hihat_filtered = signal.lfilter(b_hp, a_hp, slice_audio)

    # Snappy exponential decay
    env = np.exp(-t * 55.0)
    hihat = hihat_filtered * env

    pk = np.max(np.abs(hihat))
    if pk > 0:
        hihat = (hihat / pk) * velocity * 0.78
    return hihat.astype(np.float32)
