"""
Asynchronous In-Memory Job Manager and Background Worker
Coordinates the end-to-end procedural music generation pipeline:
  1. Preprocessing & Transcoding (FFmpeg)
  2. Complete Audio Analysis & Rule-based Classification
  3. 3-Candidate Procedural Composition & DSP Scoring
  4. Saving WAV outputs and metadata
"""

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf

from ..dsp.analysis import CompleteAnalysis, analyze_audio
from ..dsp.preprocess import PreprocessedAudio, preprocess_audio
from ..music.composer import CompositionResult, generate_candidates
from ..utils.random import generate_seed

if os.environ.get("VERCEL"):
    UPLOAD_DIR = Path("/tmp/uploads")
    OUTPUT_DIR = Path("/tmp/outputs")
else:
    UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
    OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"

try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    UPLOAD_DIR = Path("/tmp/uploads")
    OUTPUT_DIR = Path("/tmp/outputs")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class JobRecord:
    job_id: str
    status: str                       # 'queued', 'processing', 'completed', 'failed'
    stage: str                        # Current descriptive stage
    progress: int                     # 0 to 100
    created_at: float
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    source_filename: Optional[str] = None
    source_duration: Optional[float] = None
    analysis: Optional[Dict[str, Any]] = None
    winner_result: Optional[Dict[str, Any]] = None
    candidates: Optional[List[Dict[str, Any]]] = None
    beat_preference: str = "minimal"
    energy_preference: str = "balanced"
    custom_seed: Optional[int] = None
    num_candidates: Optional[int] = None


class JobManager:
    """Thread-safe in-memory job registry."""
    def __init__(self):
        self.jobs: Dict[str, JobRecord] = {}

    def create_job(
        self,
        source_filename: str,
        beat_preference: str = "minimal",
        energy_preference: str = "balanced",
        custom_seed: Optional[int] = None,
        num_candidates: Optional[int] = None
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:10]
        eff_cands = num_candidates or int(os.environ.get("NUM_CANDIDATES", "2"))
        eff_cands = max(1, min(eff_cands, 5))
        job = JobRecord(
            job_id=job_id,
            status="queued",
            stage="Job queued",
            progress=5,
            created_at=time.time(),
            source_filename=source_filename,
            beat_preference=beat_preference,
            energy_preference=energy_preference,
            custom_seed=custom_seed,
            num_candidates=eff_cands
        )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> None:
        if job_id in self.jobs:
            job = self.jobs[job_id]
            for k, v in kwargs.items():
                setattr(job, k, v)


# Global singleton manager
job_manager = JobManager()


def run_pipeline_sync(job_id: str, input_file_path: str) -> None:
    """Execute pipeline synchronously in a background thread executor."""
    job = job_manager.get_job(job_id)
    if not job:
        return

    try:
        # Step 1: Preprocess Audio
        job_manager.update_job(job_id, status="processing", stage="Decoding & preprocessing audio (FFmpeg)...", progress=15)
        prep = preprocess_audio(input_file_path)

        # Save preprocessed clean source WAV for frontend comparison
        clean_src_path = UPLOAD_DIR / f"{job_id}_clean.wav"
        sf.write(str(clean_src_path), prep.stereo.T, prep.sr)

        # Step 2: Audio Analysis & Feature Extraction
        job_manager.update_job(job_id, stage="Extracting spectral, harmonic & rhythmic DNA...", progress=35)
        analysis = analyze_audio(prep.mono, sr=prep.sr)

        # Serialize analysis for frontend DNA visualizer
        analysis_data = {
            "classification": analysis.classification,
            "confidence": analysis.confidence,
            "reasons": analysis.classification_reasons,
            "duration": analysis.duration,
            "is_silent": prep.is_silent,
            "is_clipped": prep.is_clipped,
            "rms": analysis.amplitude.rms,
            "dynamic_range_db": analysis.amplitude.dynamic_range_db,
            "envelope_curve": analysis.amplitude.envelope_curve,
            "spectral_centroid": analysis.spectral.spectral_centroid,
            "spectral_flatness": analysis.spectral.spectral_flatness,
            "dominant_frequencies": analysis.spectral.dominant_frequencies,
            "brightness": analysis.spectral.brightness,
            "tempo_bpm": analysis.rhythm.estimated_tempo,
            "has_reliable_rhythm": analysis.rhythm.has_reliable_rhythm,
            "rhythmic_density": analysis.rhythm.rhythmic_density,
            "ioi_regularity": analysis.rhythm.ioi_regularity,
            "has_reliable_pitch": analysis.pitch.has_reliable_pitch,
            "nearest_note": analysis.pitch.nearest_note_name,
            "candidate_notes": analysis.pitch.candidate_notes,
            "chroma_energy": analysis.pitch.chroma_energy,
            "noisiness": analysis.texture.noisiness,
            "harmonicity": analysis.texture.harmonicity,
            "temporal_entropy": analysis.texture.temporal_entropy
        }
        job_manager.update_job(job_id, analysis=analysis_data, source_duration=prep.duration, progress=50)

        # Step 3: Procedural Composition (candidates count configurable)
        cands_count = job.num_candidates or 2
        stage_text = f"Procedurally synthesizing {cands_count} musical candidate{'s' if cands_count > 1 else ''}..."
        job_manager.update_job(job_id, stage=stage_text, progress=60)
        winner, all_candidates = generate_candidates(
            prep=prep,
            analysis=analysis,
            base_seed=job.custom_seed,
            beat_preference=job.beat_preference,
            energy_preference=job.energy_preference,
            num_candidates=cands_count,
            on_progress=lambda pct, msg: job_manager.update_job(job_id, progress=pct, stage=msg)
        )

        job_manager.update_job(job_id, stage="Scoring candidates & mastering winner...", progress=85)

        # Step 4: Write Audio WAV Files to Output
        candidates_data = []
        for idx, cand in enumerate(all_candidates):
            is_winner = (cand.seed == winner.seed)
            filename = f"{job_id}_cand_{idx}.wav"
            out_file = OUTPUT_DIR / filename
            # Write 44.1kHz 16-bit PCM WAV (or float32)
            sf.write(str(out_file), cand.audio.T, cand.sr, subtype="PCM_16")

            cand_info = {
                "candidate_index": idx,
                "is_winner": is_winner,
                "seed": str(cand.seed),
                "duration": cand.duration,
                "tempo_bpm": cand.tempo_bpm,
                "scale_name": cand.scale.name,
                "scale_type": cand.scale.scale_type,
                "form": cand.arrangement.selected_form,
                "stems": cand.stems_info,
                "palette": getattr(cand, "palette_info", {}),
                "score": {
                    "total": cand.score.total_score,
                    "source_usage_ratio": getattr(cand.score, "source_usage_ratio", 1.0),
                    "synthetic_audio_ratio": getattr(cand.score, "synthetic_audio_ratio", 0.0),
                    "spectral_balance": cand.score.spectral_balance_score,
                    "dynamic_range": cand.score.dynamic_range_score,
                    "musical_repetition": cand.score.musical_repetition_score,
                    "transition_smoothness": cand.score.transition_score,
                    "source_integration": cand.score.source_integration_score,
                    "clipping_penalty": cand.score.clipping_penalty,
                    "harshness_penalty": cand.score.harshness_penalty,
                    "summary": cand.score.summary
                },
                "audio_url": f"/api/audio/{job_id}/{idx}"
            }
            candidates_data.append(cand_info)

        # Winner is the first candidate in candidates_data (since candidates is sorted by score)
        winner_data = candidates_data[0]

        job_manager.update_job(
            job_id,
            status="completed",
            stage="Musical composition complete!",
            progress=100,
            completed_at=time.time(),
            winner_result=winner_data,
            candidates=candidates_data
        )

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        job_manager.update_job(
            job_id,
            status="failed",
            stage="Processing failed",
            error_message=str(e),
            progress=100
        )
        print(f"Job {job_id} failed with error:\n{err}")


async def submit_generation_job(
    input_file_path: str,
    beat_preference: str = "minimal",
    energy_preference: str = "balanced",
    custom_seed: Optional[int] = None,
    num_candidates: Optional[int] = None
) -> str:
    """Create job and run in asyncio thread executor."""
    job = job_manager.create_job(
        source_filename=Path(input_file_path).name,
        beat_preference=beat_preference,
        energy_preference=energy_preference,
        custom_seed=custom_seed,
        num_candidates=num_candidates
    )

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, run_pipeline_sync, job.job_id, input_file_path)

    return job.job_id
