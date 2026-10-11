"""
make_figures.py
────────────────────────────
Re-renders the world-map figures from the saved
seed-4 arrays (no training, no re-inference):

  fig_temporal_evolution.png, fig_prediction_comparison.png,
  fig_undiscovered_hotspots.png, fig_all_model_comparison.png

Differences from visualise.py output: figures are flattened to opaque RGB PNG
(no alpha channel, no embedded Type-3 fonts), and a small cross marks the
(0 deg, 0 deg) location where ~16% of catalogue records carry placeholder
coordinates.

Usage:  python make_figures.py
"""

import io
import os

import numpy as np
from PIL import Image

import config
import visualise as V

OUT = os.path.join(config.OUTPUT_DIR, "supplementary", "figures")
os.makedirs(OUT, exist_ok=True)
config.FIG_DIR = OUT
DPI = 300

_orig_make_geo_ax = V._make_geo_ax


def _geo_ax_with_marker(*args, **kwargs):
    ax = _orig_make_geo_ax(*args, **kwargs)
    kw = dict(marker="x", color="white", markersize=5, markeredgewidth=1.6, zorder=6)
    edge = dict(marker="x", color="black", markersize=5, markeredgewidth=2.6, zorder=5)
    if V._HAS_CARTOPY and hasattr(ax, "projection"):
        kw["transform"] = edge["transform"] = V.PROJ
    ax.plot(0, 0, **edge)
    ax.plot(0, 0, **kw)
    return ax


def _save_rgb(fig, out_path, **kwargs):
    out_path = os.path.splitext(out_path)[0] + ".png"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    Image.open(buf).convert("RGB").save(out_path, optimize=True)
    print(f"Saved: {out_path}")
    return out_path


V._make_geo_ax = _geo_ax_with_marker
V._savefig_safe = _save_rgb


def all_model_comparison(gt, preds, lf):
    """Same layout as visualise.plot_all_model_comparison, from saved predictions."""
    import matplotlib.pyplot as plt
    panels = [("Ground Truth\n(2010s)", gt, "#006600"),
              ("ConvLSTM\n(Ours)", preds["convlstm"], "#0055aa"),
              ("AttentionConvLSTM\n(Ours)", preds["attention"], "#0055aa"),
              ("PredRNN\n(Ours)", preds["predrnn"], "#0055aa"),
              ("Last-Frame\n(Baseline)", lf, "#555555")]
    vmax = max(float(gt.max()), 0.01)
    fig = plt.figure(figsize=(len(panels) * 4.5, 3.2))
    gs = fig.add_gridspec(1, len(panels), wspace=0.06)
    for j, (title, data, tcol) in enumerate(panels):
        ax = V._make_geo_ax(fig, gs[0, j], title=title, title_color=tcol, fontsize=9)
        im = V._imshow_geo(ax, data, V.DENSITY_CMAP, vmin=0, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.88, pad=0.04, aspect=22)
        cb.ax.tick_params(labelsize=6)
    _save_rgb(fig, os.path.join(OUT, "fig_all_model_comparison.png"))
    plt.close(fig)


def _rename(src, dst):
    src, dst = os.path.join(OUT, src), os.path.join(OUT, dst)
    if os.path.exists(src):
        os.replace(src, dst)


if __name__ == "__main__":
    cum = np.load(os.path.join(config.MAPS_DIR, "cumulative_maps.npy"))
    inc = np.load(os.path.join(config.MAPS_DIR, "incremental_maps.npy"))
    mask = np.load(os.path.join(config.MAPS_DIR, "bias_mask.npy"))
    T = config.TEST_DECADE
    preds = {n: np.load(os.path.join(config.OUTPUT_DIR, f"test_prediction_{n}.npy"))
             for n in ("convlstm", "attention", "predrnn")}

    V.plot_temporal_evolution(cum, show_decades=[10, 13, 16, 18, 20, 21])
    _rename("fig3_temporal_evolution.png", "fig_temporal_evolution.png")
    V.plot_prediction_comparison(pred_map=preds["attention"], target_map=inc[T],
                                 input_maps=cum[T - config.SEQ_LEN:T], decade_idx=T,
                                 model_name="AttentionConvLSTM")
    _rename("fig4_prediction_comparison.png", "fig_prediction_comparison.png")
    V.plot_undiscovered_hotspots(preds["attention"], mask)
    _rename("fig5_undiscovered_hotspots.png", "fig_undiscovered_hotspots.png")
    all_model_comparison(inc[T], preds, cum[T - 1])
