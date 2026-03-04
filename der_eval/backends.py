"""Diarization backend initialisation and per-backend inference."""

import logging
import os
import time
from pathlib import Path

from pyannote.core import Annotation, Segment

log = logging.getLogger(__name__)


# ── Initialisation ────────────────────────────────────────────────────────────

def init_backend(cfg: dict) -> dict:
    """Load/connect the requested backend. Returns a context dict."""
    backend = cfg["backend"]

    if backend == "pyannote_local":
        return _init_pyannote(cfg)
    elif backend == "rttm":
        return _init_rttm(cfg)
    elif backend == "aws":
        return _init_aws(cfg)
    elif backend == "revai":
        return _init_revai(cfg)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Choose from: pyannote_local, rttm, aws, revai"
        )


def _init_pyannote(cfg: dict) -> dict:
    import torch
    from pyannote.audio import Pipeline

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    log.info("Loading pyannote pipeline on %s", device)
    pipeline = Pipeline.from_pretrained(cfg["pyannote_model"])
    pipeline.to(__import__("torch").device(device))
    log.info("pyannote pipeline ready: %s", cfg["pyannote_model"])
    return {"pipeline": pipeline, "device": device}


def _init_rttm(cfg: dict) -> dict:
    rttm_dir = Path(cfg["rttm_dir"])
    if not rttm_dir.is_dir():
        raise FileNotFoundError(f"RTTM directory not found: '{rttm_dir}'")
    rttm_map: dict[int, Path] = {}
    for p in sorted(rttm_dir.glob("*.rttm")):
        prefix = p.stem.split("_")[0]
        if prefix.isdigit():
            rttm_map[int(prefix)] = p
    if not rttm_map:
        raise FileNotFoundError(f"No .rttm files found in '{rttm_dir}'")
    log.info("RTTM backend: %d files in '%s'", len(rttm_map), rttm_dir)
    return {"rttm_map": rttm_map, "rttm_dir": rttm_dir}


def _init_aws(cfg: dict) -> dict:
    import boto3
    key    = cfg["aws_access_key_id"]     or os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = cfg["aws_secret_access_key"] or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not (key and secret and cfg["aws_s3_bucket"]):
        raise ValueError(
            "AWS backend requires aws_access_key_id, "
            "aws_secret_access_key, and aws_s3_bucket."
        )
    kw = dict(
        region_name=cfg["aws_region"],
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )
    log.info("AWS clients ready (region: %s, bucket: %s)",
             cfg["aws_region"], cfg["aws_s3_bucket"])
    return {
        "transcribe": __import__("boto3").client("transcribe", **kw),
        "s3":         __import__("boto3").client("s3",         **kw),
    }


def _init_revai(cfg: dict) -> dict:
    from rev_ai import apiclient
    token = cfg["revai_access_token"] or os.environ.get("REVAI_ACCESS_TOKEN", "")
    if not token:
        raise ValueError(
            "Rev.ai backend requires revai_access_token "
            "in config or REVAI_ACCESS_TOKEN env var."
        )
    log.info("Rev.ai client ready")
    return {"client": apiclient.RevAiAPIClient(token)}


# ── Inference ─────────────────────────────────────────────────────────────────

def diarize(wav_path: str, duration: float, row_index: int,
            cfg: dict, ctx: dict) -> Annotation:
    """Run diarization and return a pyannote Annotation."""
    backend = cfg["backend"]
    if backend == "pyannote_local":
        return _diarize_pyannote(wav_path, ctx)
    elif backend == "rttm":
        return _diarize_rttm(row_index, ctx)
    elif backend == "aws":
        return _diarize_aws(wav_path, ctx, cfg)
    elif backend == "revai":
        return _diarize_revai(wav_path, ctx)


def _diarize_pyannote(wav_path: str, ctx: dict) -> Annotation:
    raw = ctx["pipeline"](wav_path)
    annotation = getattr(raw, "speaker_diarization", raw)
    ann = Annotation()
    for seg, _, spk in annotation.itertracks(yield_label=True):
        ann[seg] = spk
    return ann


def _diarize_rttm(row_index: int, ctx: dict) -> Annotation:
    rttm_map: dict[int, Path] = ctx["rttm_map"]
    if row_index not in rttm_map:
        raise FileNotFoundError(
            f"No RTTM file for row {row_index}. "
            f"Expected {row_index:04d}_<anything>.rttm in '{ctx['rttm_dir']}'"
        )
    ann = Annotation()
    with open(rttm_map[row_index]) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) != 10 or parts[0] != "SPEAKER":
                log.warning("RTTM %s line %d skipped: unexpected format",
                            rttm_map[row_index].name, lineno)
                continue
            try:
                start = float(parts[3])
                dur   = float(parts[4])
                spk   = parts[7]
            except (ValueError, IndexError) as e:
                log.warning("RTTM %s line %d parse error: %s",
                            rttm_map[row_index].name, lineno, e)
                continue
            end = start + dur
            if end > start:
                ann[Segment(start, end)] = spk
    return ann


def _diarize_aws(wav_path: str, ctx: dict, cfg: dict) -> Annotation:
    import json
    import urllib.request

    key      = f"diar_eval/{Path(wav_path).name}"
    job_name = f"diar-eval-{Path(wav_path).stem}-{int(time.time())}"

    ctx["s3"].upload_file(wav_path, cfg["aws_s3_bucket"], key)
    ctx["transcribe"].start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": f"s3://{cfg['aws_s3_bucket']}/{key}"},
        MediaFormat="wav",
        LanguageCode="is-IS",
        Settings={"ShowSpeakerLabels": True, "MaxSpeakerLabels": 10},
    )
    while True:
        status = ctx["transcribe"].get_transcription_job(
            TranscriptionJobName=job_name
        )["TranscriptionJob"]["TranscriptionJobStatus"]
        if status == "COMPLETED":
            break
        if status == "FAILED":
            raise RuntimeError(f"AWS Transcribe job failed: {job_name}")
        time.sleep(5)

    result_uri = ctx["transcribe"].get_transcription_job(
        TranscriptionJobName=job_name
    )["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    result = json.loads(urllib.request.urlopen(result_uri).read())

    ann = Annotation()
    for seg in result["results"].get("speaker_labels", {}).get("segments", []):
        s, e = float(seg["start_time"]), float(seg["end_time"])
        if e > s:
            ann[Segment(s, e)] = seg["speaker_label"]
    return ann


def _diarize_revai(wav_path: str, ctx: dict) -> Annotation:
    job = ctx["client"].submit_job_local_file(wav_path, diarization=True)
    while True:
        status = ctx["client"].get_job_details(job.id).status.name
        if status == "TRANSCRIBED":
            break
        if status == "FAILED":
            raise RuntimeError(f"Rev.ai job failed: {job.id}")
        time.sleep(5)
    transcript = ctx["client"].get_transcript_object(job.id)
    ann = Annotation()
    for mono in transcript.monologues:
        s, e = mono.elements[0].ts, mono.elements[-1].end_ts
        if e > s:
            ann[Segment(s, e)] = str(mono.speaker)
    return ann