"""Audio materialisation — writes dataset bytes to uniquely named WAV files."""

import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def sha1_short(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()[:8]


def materialise_audio(ds, cfg: dict) -> dict[int, str]:
    """Write audio bytes from each dataset row to a WAV file on disk.

    Returns a mapping of {row_index: wav_path}.
    Skips files that already exist so runs are safely resumable.
    """
    out = Path(cfg["audio_out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    idx2path: dict[int, str] = {}
    for i in range(len(ds)):
        a = ds[i][cfg["audio_col"]]
        b = a.get("bytes")
        if not b:
            log.warning("Row %d has no audio bytes — skipped", i)
            continue
        base = Path(a.get("path") or "audio.wav").stem
        p    = out / f"{i:04d}_{base}_{sha1_short(b)}.wav"
        if not p.exists():
            p.write_bytes(b)
        idx2path[i] = str(p)

    total_sec = sum(ds[i][cfg["duration_col"]] for i in idx2path)
    log.info(
        "Audio ready: %d files  |  %.2f h (%.0f min)",
        len(idx2path), total_sec / 3600, total_sec / 60,
    )
    return idx2path