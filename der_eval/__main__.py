"""Entry point — wires all modules together."""

import logging

from datasets import Audio, load_dataset

from der_eval.audio import materialise_audio
from der_eval.backends import init_backend
from der_eval.config import parse_args
from der_eval.metrics import evaluate
from der_eval.report import plot_der_vs_duration, print_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = parse_args()

    if cfg["hf_token"]:
        from huggingface_hub import login
        login(token=cfg["hf_token"])

    log.info("Loading dataset %s  (%s)", cfg["dataset_id"], cfg["split"])
    ds = load_dataset(cfg["dataset_id"], split=cfg["split"])
    ds = ds.cast_column(cfg["audio_col"], Audio(decode=False))
    log.info("Loaded %d rows", len(ds))

    idx2path = materialise_audio(ds, cfg)
    ctx      = init_backend(cfg)
    df       = evaluate(ds, idx2path, cfg, ctx)

    print_summary(df, cfg)
    plot_der_vs_duration(df, cfg)


if __name__ == "__main__":
    main()