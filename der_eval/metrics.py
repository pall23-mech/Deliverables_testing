"""Reference annotation building and evaluation loop."""

import gc
import logging
from pathlib import Path

import pandas as pd
import torch
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm

from der_eval.backends import diarize

log = logging.getLogger(__name__)


def reference_from_segments(segments_list) -> Annotation:
    """Convert a dataset segment list to a pyannote Annotation."""
    ann = Annotation()
    for seg in segments_list:
        s   = float(seg["start"])
        e   = float(seg["end"])
        spk = seg["speaker"]
        if e > s:
            ann[Segment(s, e)] = spk
    return ann


def evaluate(ds, idx2path: dict, cfg: dict, ctx: dict) -> pd.DataFrame:
    """Run diarization + DER computation over every row in idx2path.

    Returns a DataFrame with one row per clip.
    Failures are logged and written to a separate CSV.
    """
    backend    = cfg["backend"]
    der_metric = DiarizationErrorRate()
    rows: list[dict]     = []
    failures: list[dict] = []

    for i in tqdm(sorted(idx2path), desc=f"Diarizing [{backend}]"):
        hyp = None
        try:
            wav_path = str(Path(idx2path[i]).resolve())
            if not (Path(wav_path).exists() and Path(wav_path).stat().st_size > 0):
                raise FileNotFoundError(f"Audio file missing or empty: {wav_path}")

            hyp = diarize(
                wav_path,
                ds[i][cfg["duration_col"]],
                row_index=i,
                cfg=cfg,
                ctx=ctx,
            )
            ref = reference_from_segments(ds[i][cfg["segments_col"]])
            uem = Timeline([Segment(0, ds[i][cfg["duration_col"]])])
            der = float(der_metric(ref, hyp, uem=uem))

            rows.append(dict(
                index        = i,
                backend      = backend,
                dataset      = cfg["dataset_id"],
                split        = cfg["split"],
                audio_path   = wav_path,
                duration     = ds[i][cfg["duration_col"]],
                num_segments = len(ds[i][cfg["segments_col"]]),
                der          = der,
                der_percent  = der * 100,
            ))

        except Exception as e:
            log.error("Row %d failed: %s", i, e)
            failures.append(dict(
                index      = i,
                backend    = backend,
                audio_path = idx2path.get(i),
                error      = str(e),
            ))

        finally:
            del hyp
            if backend == "pyannote_local":
                torch.cuda.empty_cache()
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(cfg["results_csv"], index=False)
    log.info("Results saved to %s", cfg["results_csv"])

    if failures:
        fail_path = cfg["results_csv"].replace(".csv", "_failures.csv")
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        log.warning("%d failures saved to %s", len(failures), fail_path)

    return df