"""Configuration defaults and CLI argument parsing."""

import argparse
from copy import deepcopy

# ── Defaults — edit this block ─────────────────────────────────────────────
DEFAULTS = dict(
    # Backend: pyannote_local | rttm | aws | revai
    backend               = "pyannote_local",

    # pyannote_local
    pyannote_model        = "pyannote/speaker-diarization-3.1",

    # rttm  — files named {index:04d}_<anything>.rttm
    rttm_dir              = "./rttm_output",

    # AWS Transcribe
    aws_access_key_id     = "",   # or env var AWS_ACCESS_KEY_ID
    aws_secret_access_key = "",   # or env var AWS_SECRET_ACCESS_KEY
    aws_region            = "us-east-1",
    aws_s3_bucket         = "",

    # Rev.ai
    revai_access_token    = "",   # or env var REVAI_ACCESS_TOKEN

    # Dataset
    dataset_id            = "palli23/Spjallromur-AB-NoOverlap-v3",
    split                 = "train",
    hf_token              = "",   # leave blank to use cached HF login

    # Dataset column names
    audio_col             = "audio",
    segments_col          = "segments",
    duration_col          = "duration",

    # Audio
    audio_out_dir         = "diar_eval_audio",

    # Output
    results_csv           = "diarization_eval_results.csv",
    plot_png              = "der_vs_duration.png",
)
# ───────────────────────────────────────────────────────────────────────────


def parse_args(defaults: dict = DEFAULTS) -> dict:
    """Merge CLI arguments on top of defaults and return a config dict."""
    cfg = deepcopy(defaults)
    p = argparse.ArgumentParser(
        description="Offline DER evaluation harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend",    default=cfg["backend"],
                   choices=["pyannote_local", "rttm", "aws", "revai"])
    p.add_argument("--dataset",    default=cfg["dataset_id"],
                   metavar="HF_DATASET_ID")
    p.add_argument("--split",      default=cfg["split"],
                   help='e.g. "train" or "train[:50]"')
    p.add_argument("--hf-token",   default=cfg["hf_token"],
                   metavar="TOKEN")
    p.add_argument("--rttm-dir",   default=cfg["rttm_dir"],
                   metavar="DIR")
    p.add_argument("--audio-dir",  default=cfg["audio_out_dir"],
                   metavar="DIR")
    p.add_argument("--results",    default=cfg["results_csv"],
                   metavar="CSV")
    p.add_argument("--plot",       default=cfg["plot_png"],
                   metavar="PNG")

    args = p.parse_args()
    cfg.update(dict(
        backend       = args.backend,
        dataset_id    = args.dataset,
        split         = args.split,
        hf_token      = args.hf_token,
        rttm_dir      = args.rttm_dir,
        audio_out_dir = args.audio_dir,
        results_csv   = args.results,
        plot_png      = args.plot,
    ))
    return cfg