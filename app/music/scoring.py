"""
Objective DSP and Statistical Candidate Scoring Engine
TheUnnecessaryFM Section 19 Requirements:
  - source_usage_ratio (target: 90-100%)
  - synthetic_audio_ratio (target: 0-10%, allows 0%)
  - dynamic_range
  - repetition_score
  - source_similarity
  - clipping penalty
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import librosa
import numpy as np


@dataclass
class CandidateScore:
    total_score: float                # Final weighted score [0 - 100]
    source_usage_ratio: float         # Ratio of source-derived material [0.0 - 1.0] (Target 0.90 - 1.00)
    synthetic_audio_ratio: float      # Ratio of synthetic material [0.0 - 1.0] (Target 0.00 - 0.10)
    spectral_balance_score: float     # [0 - 25]
    dynamic_range_score: float        # [0 - 20]
    musical_repetition_score: float   # [0 - 20]
    transition_score: float           # [0 - 15]
    source_integration_score: float   # [0 - 20]
    clipping_penalty: float           # Subtraction
    harshness_penalty: float          # Subtraction
    summary: str


def score_composition(
    output_audio: np.ndarray,
    source_audio: np.ndarray,
    source_usage_ratio: float = 1.0,
    synthetic_audio_ratio: float = 0.0,
    sr: int = 44100
) -> CandidateScore:
    """
    Performs 100% objective, statistical & DSP evaluation of candidate piece.
    """
    # Convert stereo output to mono for analysis
    if len(output_audio.shape) > 1:
        out_mono = 0.5 * (output_audio[0] + output_audio[1])
    else:
        out_mono = output_audio

    # 1. Clipping Check & Penalty
    abs_out = np.abs(out_mono)
    clipped_count = int(np.sum(abs_out >= 0.999))
    clipping_penalty = float(clipped_count * 2.5)

    # 2. Harshness Penalty
    # Energy in 8kHz - 20kHz vs 1kHz - 8kHz
    sample_len = min(len(out_mono), sr * 12)
    fft_out = np.abs(np.fft.rfft(out_mono[:sample_len]))
    freqs = np.fft.rfftfreq(sample_len, 1.0 / sr)

    harsh_mask = freqs >= 8000
    mid_mask = (freqs >= 1000) & (freqs < 8000)
    harsh_pwr = np.sum(fft_out[harsh_mask] ** 2) if np.any(harsh_mask) else 1e-6
    mid_pwr = np.sum(fft_out[mid_mask] ** 2) if np.any(mid_mask) else 1e-6
    harsh_ratio = harsh_pwr / max(1e-6, mid_pwr)
    harshness_penalty = float(np.clip((harsh_ratio - 0.45) * 18.0, 0.0, 15.0))

    # 3. Spectral Balance Score
    sub_mask = (freqs >= 30) & (freqs < 120)
    bass_mask = (freqs >= 120) & (freqs < 350)
    sub_pwr = np.sum(fft_out[sub_mask] ** 2) if np.any(sub_mask) else 1e-6
    bass_pwr = np.sum(fft_out[bass_mask] ** 2) if np.any(bass_mask) else 1e-6

    total_pwr = sub_pwr + bass_pwr + mid_pwr + harsh_pwr + 1e-9
    sub_ratio = sub_pwr / total_pwr
    mid_ratio = mid_pwr / total_pwr

    sub_bal = 1.0 - abs(sub_ratio - 0.20) * 2.5
    mid_bal = 1.0 - abs(mid_ratio - 0.50) * 2.0
    spec_bal = float(np.clip((sub_bal * 0.5 + mid_bal * 0.5), 0.0, 1.0))
    spectral_balance_score = round(spec_bal * 25.0, 1)

    # 4. Dynamic Range Score
    rms = np.sqrt(np.mean(out_mono ** 2))
    pk = np.max(abs_out)
    crest = pk / max(rms, 1e-4)
    dyn_fitness = 1.0 - abs(crest - 4.5) * 0.20
    dynamic_range_score = round(float(np.clip(dyn_fitness, 0.25, 1.0)) * 20.0, 1)

    # 5. Musical Repetition vs Variety Score (autocorrelation of energy envelope)
    hop = 2048
    rms_env = librosa.feature.rms(y=out_mono, hop_length=hop)[0]
    if len(rms_env) > 10:
        ac = librosa.autocorrelate(rms_env, max_size=min(len(rms_env), int(sr / hop * 6)))
        if len(ac) > 2:
            ac_norm = ac[1:] / (ac[0] + 1e-6)
            peak_periodicity = float(np.max(ac_norm))
            rep_fit = 1.0 - abs(peak_periodicity - 0.50) * 2.0
            musical_repetition_score = round(float(np.clip(rep_fit, 0.3, 1.0)) * 20.0, 1)
        else:
            musical_repetition_score = 15.0
    else:
        musical_repetition_score = 15.0

    # 6. Section Transition Smoothness Score
    diffs = np.abs(np.diff(rms_env))
    max_jump = np.max(diffs) if len(diffs) > 0 else 0.0
    mean_val = np.mean(rms_env) if len(rms_env) > 0 else 1.0
    jump_ratio = max_jump / max(mean_val, 1e-4)
    transition_score = round(float(np.clip(1.0 - (jump_ratio / 3.0), 0.25, 1.0)) * 15.0, 1)

    # 7. Source Integration & Similarity Score
    src_mono = 0.5 * (source_audio[0] + source_audio[1]) if len(source_audio.shape) > 1 else source_audio
    src_len = min(len(src_mono), sr * 12)
    src_fft = np.abs(np.fft.rfft(src_mono[:src_len]))
    comp_len = min(len(fft_out), len(src_fft))

    if comp_len > 100:
        v_out = fft_out[:comp_len] / (np.linalg.norm(fft_out[:comp_len]) + 1e-9)
        v_src = src_fft[:comp_len] / (np.linalg.norm(src_fft[:comp_len]) + 1e-9)
        sim = float(np.dot(v_out, v_src))
        int_score = float(np.clip(sim * 2.0 + 0.4, 0.35, 1.0))
        source_integration_score = round(int_score * 20.0, 1)
    else:
        source_integration_score = 16.0

    # Source usage bonus / synthetic penalty (Section 19: 90-100% source, 0-10% synth)
    source_bonus = 5.0 * (source_usage_ratio - 0.90) / 0.10 if source_usage_ratio >= 0.90 else -15.0
    synthetic_penalty = max(0.0, (synthetic_audio_ratio - 0.05) * 40.0)

    # Total Score
    raw_total = (
        spectral_balance_score +
        dynamic_range_score +
        musical_repetition_score +
        transition_score +
        source_integration_score +
        source_bonus -
        clipping_penalty -
        harshness_penalty -
        synthetic_penalty
    )
    total_score = round(float(np.clip(raw_total, 0.0, 100.0)), 1)

    summary = (
        f"Score: {total_score}/100 "
        f"[SourceRatio: {int(source_usage_ratio*100)}%, "
        f"SpecBal: {spectral_balance_score}, "
        f"Dyn: {dynamic_range_score}, "
        f"Motif: {musical_repetition_score}, "
        f"SourceInt: {source_integration_score}]"
    )

    return CandidateScore(
        total_score=total_score,
        source_usage_ratio=round(source_usage_ratio, 2),
        synthetic_audio_ratio=round(synthetic_audio_ratio, 2),
        spectral_balance_score=spectral_balance_score,
        dynamic_range_score=dynamic_range_score,
        musical_repetition_score=musical_repetition_score,
        transition_score=transition_score,
        source_integration_score=source_integration_score,
        clipping_penalty=round(clipping_penalty, 1),
        harshness_penalty=round(harshness_penalty, 1),
        summary=summary
    )
