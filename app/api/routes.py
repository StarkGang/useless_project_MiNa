"""
FastAPI REST API Routes for TheUnnecessaryFM
Endpoints:
  - POST /api/analyze: Instant source feature extraction & classification
  - POST /api/generate: Background composition job submission
  - GET  /api/status/{job_id}: Real-time progress and stage tracking
  - GET  /api/result/{job_id}: Full result payload with DNA & candidate details
  - GET  /api/audio/{job_id}/{candidate_id}: Audio file streaming
  - GET  /api/source-audio/{job_id}: Preprocessed source audio streaming
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..dsp.analysis import analyze_audio
from ..dsp.preprocess import preprocess_audio
from .jobs import OUTPUT_DIR, UPLOAD_DIR, job_manager, submit_generation_job

router = APIRouter(prefix="/api")


@router.post("/analyze")
async def analyze_uploaded_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file (uploaded or recorded from mic),
    preprocesses it, and returns the complete DSP feature analysis and rule-based classification.
    """
    suffix = Path(file.filename or "recording.wav").suffix or ".wav"
    temp_filename = f"temp_analysis_{uuid.uuid4().hex[:8]}{suffix}"
    temp_path = UPLOAD_DIR / temp_filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess and analyze
        prep = preprocess_audio(str(temp_path))
        analysis = analyze_audio(prep.mono, sr=prep.sr)

        return {
            "status": "success",
            "filename": file.filename,
            "duration": analysis.duration,
            "classification": analysis.classification,
            "confidence": analysis.confidence,
            "reasons": analysis.classification_reasons,
            "is_silent": prep.is_silent,
            "is_clipped": prep.is_clipped,
            "dna": {
                "rhythmic_density": analysis.rhythm.rhythmic_density,
                "tempo_bpm": analysis.rhythm.estimated_tempo,
                "has_reliable_rhythm": analysis.rhythm.has_reliable_rhythm,
                "ioi_regularity": analysis.rhythm.ioi_regularity,
                "harmonicity": analysis.texture.harmonicity,
                "noisiness": analysis.texture.noisiness,
                "spectral_centroid": analysis.spectral.spectral_centroid,
                "brightness": analysis.spectral.brightness,
                "spectral_flatness": analysis.spectral.spectral_flatness,
                "has_reliable_pitch": analysis.pitch.has_reliable_pitch,
                "nearest_note": analysis.pitch.nearest_note_name,
                "candidate_notes": analysis.pitch.candidate_notes,
                "rms": analysis.amplitude.rms,
                "dynamic_range_db": analysis.amplitude.dynamic_range_db,
                "envelope_curve": analysis.amplitude.envelope_curve
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio analysis failed: {str(e)}")
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass


@router.post("/generate")
async def generate_music(
    file: Optional[UploadFile] = File(None),
    existing_job_id: Optional[str] = Form(None),
    beat_preference: str = Form("minimal"),
    energy_preference: str = Form("balanced"),
    seed: Optional[str] = Form(None)
):
    """
    Submits a procedural music generation request.
    Can accept a newly uploaded file/mic recording OR reuse a previously uploaded source file.
    Runs asynchronously and returns a job_id for polling.
    """
    custom_seed_val = None
    if seed and seed.strip():
        try:
            custom_seed_val = int(seed.strip())
        except ValueError:
            custom_seed_val = None

    if file is not None and file.filename:
        # Save file to uploads directory
        suffix = Path(file.filename).suffix or ".wav"
        save_filename = f"upload_{uuid.uuid4().hex[:10]}{suffix}"
        target_path = UPLOAD_DIR / save_filename

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    elif existing_job_id:
        # Check if previous job's source file exists
        source_clean = UPLOAD_DIR / f"{existing_job_id}_clean.wav"
        if not source_clean.exists():
            raise HTTPException(status_code=404, detail="Source audio for this job ID not found.")
        target_path = source_clean
    else:
        raise HTTPException(status_code=400, detail="Please upload an audio file or microphone recording.")

    # Submit background generation task
    job_id = await submit_generation_job(
        input_file_path=str(target_path),
        beat_preference=beat_preference,
        energy_preference=energy_preference,
        custom_seed=custom_seed_val
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Procedural music generation job queued."
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Poll progress and current processing stage."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "error": job.error_message
    }


@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    """Retrieve complete metadata and candidate links when job is completed."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.status == "failed":
        return JSONResponse(status_code=500, content={"status": "failed", "error": job.error_message})

    if job.status != "completed":
        return {
            "status": job.status,
            "progress": job.progress,
            "stage": job.stage
        }

    return {
        "status": "completed",
        "job_id": job.job_id,
        "source": {
            "filename": job.source_filename,
            "duration": job.source_duration,
            "audio_url": f"/api/source-audio/{job.job_id}",
            "analysis": job.analysis
        },
        "winner": job.winner_result,
        "candidates": job.candidates,
        "settings": {
            "beat_preference": job.beat_preference,
            "energy_preference": job.energy_preference,
            "custom_seed": job.custom_seed
        }
    }


@router.get("/audio/{job_id}/{candidate_id}")
async def stream_candidate_audio(job_id: str, candidate_id: int):
    """Stream generated candidate WAV file."""
    filename = f"{job_id}_cand_{candidate_id}.wav"
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=f"unnecessaryfm_{job_id}_cand{candidate_id}.wav"
    )


@router.get("/source-audio/{job_id}")
async def stream_source_audio(job_id: str):
    """Stream preprocessed clean source WAV file."""
    file_path = UPLOAD_DIR / f"{job_id}_clean.wav"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Source audio file not found.")

    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=f"source_{job_id}.wav"
    )
