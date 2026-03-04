"""Console summary and DER-vs-duration scatter plot."""

import logging

import pandas as pd
from matplotlib import pyplot as plt

log = logging.getLogger(__name__)


def print_summary(df: pd.DataFrame, cfg: dict) -> None:
    """Print aggregated DER statistics to stdout."""
    ok = df["der"].notna() if not df.empty else pd.Series([], dtype=bool)

    print("\n" + "=" * 45)
    print("  Diarization Evaluation Summary")
    print("=" * 45)
    print(f"  Backend   : {cfg['backend']}")
    print(f"  Dataset   : {cfg['dataset_id']}  ({cfg['split']})")
    print(f"  Processed : {len(df)}  |  DER computed: {ok.sum()}")

    if ok.any():
        avg_der = df.loc[ok, "der"].mean()
        w_der   = (
            (df.loc[ok, "der"] * df.loc[ok, "duration"]).sum()
            / df.loc[ok, "duration"].sum()
        )
        std_der = df.loc[ok, "der"].std()
        print(f"  Avg DER (unweighted)   : {avg_der*100:.2f}%")
        print(f"  Avg DER (dur-weighted) : {w_der*100:.2f}%")
        print(f"  Std deviation          : {std_der*100:.2f}%")
        print(f"  Results CSV : {cfg['results_csv']}")
    else:
        print("  No DER values computed — check failures CSV.")

    print("=" * 45 + "\n")


def plot_der_vs_duration(df: pd.DataFrame, cfg: dict) -> None:
    """Save and show a DER-vs-duration scatter plot."""
    ok = df["der"].notna() if not df.empty else pd.Series([], dtype=bool)
    if not ok.any():
        log.warning("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        df.loc[ok, "duration"],
        df.loc[ok, "der_percent"],
        s=32, alpha=0.7,
    )
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("DER (%)")
    ax.set_title(f"DER vs clip duration  [{cfg['backend']}]")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(cfg["plot_png"], dpi=150)
    log.info("Plot saved to %s", cfg["plot_png"])
    plt.show()