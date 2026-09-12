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

is_serverless = bool(
    os.environ.get("VERCEL")
    or os.environ.get("VERCEL_ENV")
    or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    or os.environ.get("USE_TMP_STORAGE")
)

if is_serverless:
    UPLOAD_DIR = Path("/tmp/uploads")
    OUTPUT_DIR = Path("/tmp/outputs")
else:
    try:
        candidate_upload = Path(__file__).resolve().parent.parent.parent / "uploads"
        candidate_output = Path(__file__).resolve().parent.parent.parent / "outputs"
        candidate_upload.mkdir(parents=True, exist_ok=True)
        candidate_output.mkdir(parents=True, exist_ok=True)
        # Test write permission
        test_file = candidate_upload / ".test_write"
        test_file.touch()
        test_file.unlink()
        UPLOAD_DIR = candidate_upload
        OUTPUT_DIR = candidate_output
    except Exception:
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
    beat_preference: str = "pop"
    energy_preference: str = "low"
    custom_seed: Optional[int] = None
    num_candidates: Optional[int] = None
    target_duration: float = 30.0
    vocal_mode: str = "auto"
    played_artists: Optional[List[str]] = None
    preferred_artist_id: Optional[str] = None



class JobManager:
    """Thread-safe and disk-backed job registry with real-time SSE pub/sub streaming and disk pruning."""
    def __init__(self):
        self.jobs: Dict[str, JobRecord] = {}
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def subscribe(self, job_id: str) -> asyncio.Queue:
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        q = asyncio.Queue()
        if job_id not in self.subscribers:
            self.subscribers[job_id] = []
        self.subscribers[job_id].append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        if job_id in self.subscribers:
            try:
                self.subscribers[job_id].remove(q)
            except ValueError:
                pass
            if not self.subscribers[job_id]:
                del self.subscribers[job_id]

    def cleanup_old_files(self, max_age_seconds: float = 1800.0, max_jobs: int = 15) -> None:
        """Prune old audio files and keep memory / disk lean on Render free tier."""
        now = time.time()
        # 1. Prune memory jobs if exceeding max_jobs
        if len(self.jobs) > max_jobs:
            sorted_jobs = sorted(self.jobs.values(), key=lambda j: j.created_at)
            for old_j in sorted_jobs[:-max_jobs]:
                self.jobs.pop(old_j.job_id, None)

        # 2. Prune old temporary audio files on disk (> 30 minutes old)
        for folder in [UPLOAD_DIR, OUTPUT_DIR]:
            try:
                for item in folder.glob("*"):
                    if item.is_file() and not item.name.startswith("test_"):
                        try:
                            mtime = item.stat().st_mtime
                            if (now - mtime) > max_age_seconds:
                                item.unlink(missing_ok=True)
                        except Exception:
                            pass
            except Exception:
                pass

    def _save_to_disk(self, job: JobRecord) -> None:
        try:
            record_path = OUTPUT_DIR / f"{job.job_id}_record.json"
            data = asdict(job)
            import json
            with open(record_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_from_disk(self, job_id: str) -> Optional[JobRecord]:
        try:
            record_path = OUTPUT_DIR / f"{job_id}_record.json"
            if record_path.exists():
                import json
                with open(record_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                job = JobRecord(**data)
                self.jobs[job_id] = job
                return job
        except Exception:
            pass
        return None

    def create_job(
        self,
        source_filename: str,
        beat_preference: str = "pop",
        energy_preference: str = "low",
        custom_seed: Optional[int] = None,
        num_candidates: Optional[int] = None,
        target_duration: float = 30.0,
        vocal_mode: str = "auto",
        played_artists: Optional[List[str]] = None,
        preferred_artist_id: Optional[str] = None
    ) -> JobRecord:
        # Prune old files on each new job creation
        self.cleanup_old_files()

        job_id = uuid.uuid4().hex[:10]
        eff_cands = num_candidates or int(os.environ.get("NUM_CANDIDATES", "1"))
        eff_cands = max(1, min(eff_cands, 3))
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
            num_candidates=eff_cands,
            target_duration=target_duration,
            vocal_mode=vocal_mode,
            played_artists=played_artists,
            preferred_artist_id=preferred_artist_id
        )
        self.jobs[job_id] = job
        self._save_to_disk(job)
        return job


    def get_job(self, job_id: str) -> Optional[JobRecord]:
        job = self.jobs.get(job_id)
        if not job:
            job = self._load_from_disk(job_id)
        return job

    def update_job(self, job_id: str, **kwargs) -> None:
        job = self.get_job(job_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
            self._save_to_disk(job)

            # Notify active SSE subscribers immediately
            if job_id in self.subscribers:
                msg = {
                    "job_id": job.job_id,
                    "status": job.status,
                    "stage": job.stage,
                    "progress": job.progress,
                    "error": job.error_message
                }
                for q in list(self.subscribers[job_id]):
                    try:
                        if self.loop and self.loop.is_running():
                            self.loop.call_soon_threadsafe(q.put_nowait, msg)
                        else:
                            q.put_nowait(msg)
                    except Exception:
                        pass


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
        job_manager.update_job(job_id, stage="Extracting spectral, harmonic & rhythmic DNA...", progress=20)
        progress_cb = lambda pct, msg: job_manager.update_job(job_id, progress=pct, stage=msg)
        analysis = analyze_audio(prep.mono, sr=prep.sr, on_progress=progress_cb)

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
            "temporal_entropy": analysis.texture.temporal_entropy,
            "smart_profile": {
                "primary_category": analysis.smart_profile.primary_category,
                "display_title": analysis.smart_profile.display_title,
                "detected_elements": analysis.smart_profile.detected_elements,
                "has_speech_or_vocal": analysis.smart_profile.has_speech_or_vocal,
                "has_humming": analysis.smart_profile.has_humming,
                "has_beatbox": analysis.smart_profile.has_beatbox,
                "has_traffic_or_engine": analysis.smart_profile.has_traffic_or_engine,
                "has_foley_percussive": analysis.smart_profile.has_foley_percussive,
                "has_ambient_bed": analysis.smart_profile.has_ambient_bed,
                "autotune_enabled": analysis.smart_profile.autotune_enabled,
                "autotune_strength": analysis.smart_profile.autotune_strength,
                "settings_summary": analysis.smart_profile.smart_settings_summary,
                "recommended_styles": analysis.smart_profile.recommended_styles
            } if getattr(analysis, 'smart_profile', None) else None
        }
        job_manager.update_job(job_id, analysis=analysis_data, source_duration=prep.duration, progress=52)

        # Step 3: Procedural Composition (candidates count configurable)
        cands_count = job.num_candidates or 1
        winner, all_candidates = generate_candidates(
            prep=prep,
            analysis=analysis,
            base_seed=job.custom_seed,
            beat_preference=job.beat_preference,
            energy_preference=job.energy_preference,
            vocal_mode=getattr(job, 'vocal_mode', 'auto'),
            num_candidates=cands_count,
            on_progress=progress_cb,
            target_duration=job.target_duration,
            played_artists=job.played_artists,
            preferred_artist_id=job.preferred_artist_id
        )

        job_manager.update_job(job_id, stage="Finalizing audio stems & exporting WAV...", progress=98)

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
                "artist_name": cand.artist_name,
                "artist_id": cand.artist_id,
                "artist_track_hint": cand.artist_track_hint,
                "style_name": getattr(cand, "style_name", cand.artist_name),
                "style_desc": getattr(cand, "style_desc", cand.artist_track_hint),
                "cycle_reset": getattr(cand, "cycle_reset", False),
                "stems": cand.stems_info,
                "palette": getattr(cand, "palette_info", {}),
                "smart_profile": {
                    "display_title": cand.smart_profile.display_title,
                    "detected_elements": cand.smart_profile.detected_elements,
                    "autotune_enabled": cand.smart_profile.autotune_enabled,
                    "settings_summary": cand.smart_profile.smart_settings_summary
                } if getattr(cand, "smart_profile", None) else None,
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
        import gc
        gc.collect()

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
    beat_preference: str = "pop",
    energy_preference: str = "low",
    custom_seed: Optional[int] = None,
    num_candidates: Optional[int] = None,
    target_duration: float = 30.0,
    vocal_mode: str = "auto",
    played_artists: Optional[List[str]] = None,
    preferred_artist_id: Optional[str] = None
) -> str:
    """Create job and run in asyncio thread executor."""
    job = job_manager.create_job(
        source_filename=Path(input_file_path).name,
        beat_preference=beat_preference,
        energy_preference=energy_preference,
        custom_seed=custom_seed,
        num_candidates=num_candidates,
        target_duration=target_duration,
        vocal_mode=vocal_mode,
        played_artists=played_artists,
        preferred_artist_id=preferred_artist_id
    )


    loop = asyncio.get_running_loop()
    if is_serverless:
        # On Vercel / serverless: background threads are frozen immediately when the HTTP response returns!
        # Await pipeline execution so generation completes reliably within the Lambda invocation.
        await loop.run_in_executor(None, run_pipeline_sync, job.job_id, input_file_path)
    else:
        loop.run_in_executor(None, run_pipeline_sync, job.job_id, input_file_path)

    return job.job_id

