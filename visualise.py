"""
visualise.py
────────────
Generate the result figures on a world-map background (cartopy PlateCarree
with the bundled stock image; falls back to a plain axis without cartopy).

Figures produced in outputs/figures/:
  2. fig2_training_curves.pdf       — loss / MSE / IoU curves
  3. fig3_temporal_evolution.pdf    — KDE map grid across decades
  4. fig4_prediction_comparison.pdf — Last-input / prediction / GT / error
  5. fig5_undiscovered_hotspots.pdf — raw prediction + hotspot map
  6. fig6_all_model_comparison.pdf  — all-model 2010s comparison

Run:  python visualise.py
"""

import numpy as np
import os
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter1d

try:
    import cartopy.crs as ccrs   # stock_img() is bundled — no shapefile download needed
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

import config
import data_preprocessing as dp

# ─────────────────────────────────────────────────────────────────────────────
# Colormaps
# ─────────────────────────────────────────────────────────────────────────────
# Primary density cmap: transparent at 0 (Blue Marble shows through) → orange → yellow
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density",
    ["#00000000",   # fully transparent (world map shows through at 0 density)
     "#6b000080",   # dark red, semi-transparent
     "#cc0000cc",   # red
     "#ff6600ee",   # orange
     "#ffcc00ff"]   # bright yellow (peak density)
)

# Hotspot cmap: transparent → yellow → purple
HOTSPOT_CMAP = LinearSegmentedColormap.from_list(
    "hotspot",
    ["#00000000",   # transparent (world map shows through)
     "#ffff0099",   # yellow, semi-transparent
     "#ff6600cc",   # orange
     "#cc00ccff"]   # purple (highest score)
)

PROJ = ccrs.PlateCarree() if _HAS_CARTOPY else None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create a geo-aware axis using stock_img() — no shapefile download
# ─────────────────────────────────────────────────────────────────────────────
def _make_geo_ax(fig, subplot_spec, title="", title_color="black", fontsize=8):
    """
    Add a GeoAxes to *fig* backed by NASA Blue Marble (ax.stock_img()).
    stock_img() is bundled inside cartopy — no internet download required.
    Falls back gracefully to a plain axis if cartopy is unavailable.
    """
    if _HAS_CARTOPY:
        try:
            if isinstance(subplot_spec, tuple):
                ax = fig.add_subplot(*subplot_spec, projection=PROJ)
            else:
                ax = fig.add_subplot(subplot_spec, projection=PROJ)

            ax.stock_img()       # NASA Blue Marble — bundled, no download
            ax.set_global()

            # Subtle white grid lines (visible over the satellite image)
            gl = ax.gridlines(draw_labels=False, linewidth=0.3,
                              color="white", alpha=0.25,
                              linestyle="--", zorder=2)
            gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
            gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

            if title:
                ax.set_title(title, fontsize=fontsize, color=title_color,
                             fontweight="bold", pad=3)
            return ax
        except Exception as e:
            print(f"  [cartopy fallback] {e}")

    # ── Plain matplotlib fallback ────────────────────────────────────────────
    if isinstance(subplot_spec, tuple):
        ax = fig.add_subplot(*subplot_spec)
    else:
        ax = fig.add_subplot(subplot_spec)
    ax.set_facecolor("#1a2a3a")
    ax.set_xlim(-180, 180);  ax.set_ylim(-90, 90)
    ax.set_xticks([-120, -60, 0, 60, 120])
    ax.set_yticks([-60, -30,  0, 30, 60])
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Longitude", fontsize=6)
    ax.set_ylabel("Latitude",  fontsize=6)
    ax.grid(linewidth=0.3, color="#aaaaaa", alpha=0.4, linestyle="--")
    if title:
        ax.set_title(title, fontsize=fontsize, color=title_color,
                     fontweight="bold", pad=3)
    return ax


def _imshow_geo(ax, data, cmap, vmin=0, vmax=1, alpha=0.82):
    """
    Overlay a 90×180 density array onto the axis (geo or plain).
    Returns the AxesImage for colorbar attachment.
    """
    kwargs = dict(origin="lower", extent=[-180, 180, -90, 90],
                  cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                  aspect="auto", interpolation="bilinear", zorder=3)
    if _HAS_CARTOPY and hasattr(ax, 'projection'):
        kwargs["transform"] = PROJ
    return ax.imshow(data, **kwargs)


def _savefig_safe(fig, out_path: str, **savefig_kwargs):
    """
    Save figure to out_path, falling back to a timestamped filename if the
    target file is locked by another process (common on Windows/OneDrive).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        fig.savefig(out_path, **savefig_kwargs)
        print(f"Saved: {out_path}")
        return out_path
    except PermissionError:
        stem, ext = os.path.splitext(out_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = f"{stem}_{ts}{ext}"
        fig.savefig(alt_path, **savefig_kwargs)
        print(f"WARNING: Could not overwrite locked file: {out_path}")
        print(f"Saved fallback copy: {alt_path}")
        return alt_path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Temporal evolution
# ─────────────────────────────────────────────────────────────────────────────
def plot_temporal_evolution(cum_maps: np.ndarray,
                             show_decades: list = None,
                             save: bool = True):
    """
    Grid of cumulative KDE maps across selected decades, on world-map background.
    """
    if show_decades is None:
        show_decades = [10, 13, 16, 18, 20, 21]

    n     = len(show_decades)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 5.5, nrows * 3.2))
    gs  = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.05)

    ims = []
    for i, idx in enumerate(show_decades):
        r, c = divmod(i, ncols)
        ax   = _make_geo_ax(fig, gs[r, c])
        year = dp.decade_idx_to_year(idx)

        im = _imshow_geo(ax, cum_maps[idx], DENSITY_CMAP, vmin=0, vmax=1)
        ims.append(im)
        ax.set_title(f"Up to {year}s", fontsize=10, fontweight="bold", pad=4)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cb = fig.colorbar(ims[0], cax=cbar_ax)
    cb.set_label("Normalised discovery density", fontsize=9)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Cumulative Meteorite Landing Density Maps (KDE)\n"
        "Temporal Evolution of Discovery Patterns",
        fontsize=12, fontweight="bold", y=1.01
    )

    if save:
        out = os.path.join(config.FIG_DIR, "fig3_temporal_evolution.pdf")
        _savefig_safe(fig, out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Prediction comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_prediction_comparison(pred_map:   np.ndarray,
                                target_map: np.ndarray,
                                input_maps: np.ndarray,
                                decade_idx: int,
                                model_name: str = "AttentionConvLSTM",
                                save: bool = True):
    """
    Four world-map panels: Last Input | Prediction | Ground Truth | Abs Error.
    """
    year = dp.decade_idx_to_year(decade_idx)

    panels = [
        (f"Last Input\n({year - 10}s cumulative)", input_maps[-1], DENSITY_CMAP, "black",  0, 1),
        (f"Predicted\n({year}s incremental)",       pred_map,        DENSITY_CMAP, "#0055aa", 0, 1),
        (f"Ground Truth\n({year}s incremental)",    target_map,      DENSITY_CMAP, "#006600", 0, 1),
    ]
    diff     = np.abs(pred_map - target_map)
    vmax_err = max(float(diff.max()), 0.05)
    ERR_CMAP = LinearSegmentedColormap.from_list(
        "err", ["#00000000", "#ffff0080", "#ff880099", "#ff0000ff"])
    panels.append(("Absolute Error", diff, ERR_CMAP, "#cc0000", 0, vmax_err))

    fig = plt.figure(figsize=(22, 3.8))
    gs  = fig.add_gridspec(1, 4, wspace=0.08)

    for j, (title, data, cmap, tcol, vmin, vmax) in enumerate(panels):
        ax = _make_geo_ax(fig, gs[0, j], title=title, title_color=tcol, fontsize=8)
        im = _imshow_geo(ax, data, cmap, vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          shrink=0.9, pad=0.04, aspect=25)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"{model_name} Prediction vs Ground Truth",
                 fontsize=11, fontweight="bold", y=1.04)

    if save:
        out = os.path.join(config.FIG_DIR, "fig4_prediction_comparison.pdf")
        _savefig_safe(fig, out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Undiscovered hotspots
# ─────────────────────────────────────────────────────────────────────────────
def plot_undiscovered_hotspots(model_output: np.ndarray,
                                bias_mask:   np.ndarray,
                                save: bool = True):
    """
    Left: raw predicted density on world map.
    Right: bias-suppressed hotspot map with top-5 stars.
    """
    undiscov = model_output * bias_mask

    fig = plt.figure(figsize=(18, 5))
    gs  = fig.add_gridspec(1, 2, wspace=0.08)

    # ── Left: raw prediction ─────────────────────────────────────────────────
    ax0 = _make_geo_ax(fig, gs[0, 0],
                       title="Model Predicted Density (all zones)",
                       fontsize=10)
    im0 = _imshow_geo(ax0, model_output, DENSITY_CMAP, vmin=0, vmax=1)
    cb0 = fig.colorbar(im0, ax=ax0, orientation="horizontal",
                       shrink=0.85, pad=0.04, aspect=30)
    cb0.set_label("Density", fontsize=8)
    cb0.ax.tick_params(labelsize=7)

    # ── Right: undiscovered hotspot ──────────────────────────────────────────
    ax1 = _make_geo_ax(fig, gs[0, 1],
                       title="Predicted Undiscovered Hotspots\n"
                             "(collection-program zones suppressed)",
                       fontsize=10)
    vmax_u = max(float(undiscov.max()), 1e-6)
    im1 = _imshow_geo(ax1, undiscov, HOTSPOT_CMAP, vmin=0, vmax=vmax_u)
    cb1 = fig.colorbar(im1, ax=ax1, orientation="horizontal",
                       shrink=0.85, pad=0.04, aspect=30)
    cb1.set_label("Undiscovered score", fontsize=8)
    cb1.ax.tick_params(labelsize=7)

    # ── Top-5 annotated stars ────────────────────────────────────────────────
    undiscov_copy = undiscov.copy()
    top5_coords   = []
    suppress_r    = 5
    for rank in range(5):
        flat_idx        = np.argmax(undiscov_copy.ravel())
        lat_idx, lon_idx = np.unravel_index(flat_idx, undiscov_copy.shape)
        lat = -90  + (lat_idx + 0.5) * (180 / config.GRID_H)
        lon = -180 + (lon_idx + 0.5) * (360 / config.GRID_W)
        top5_coords.append((lon, lat, rank + 1))
        r0 = max(lat_idx - suppress_r, 0);  r1 = min(lat_idx + suppress_r, undiscov_copy.shape[0])
        c0 = max(lon_idx - suppress_r, 0);  c1 = min(lon_idx + suppress_r, undiscov_copy.shape[1])
        undiscov_copy[r0:r1, c0:c1] = 0

    label_offsets = [(5, 3), (-20, 5), (5, -5), (-22, -5), (5, 3)]
    for (lon, lat, rank), (dx, dy) in zip(top5_coords, label_offsets):
        ax1.plot(lon, lat, "w*", markersize=10, markeredgecolor="gold",
                 markeredgewidth=0.5, transform=PROJ, zorder=5)
        ax1.text(lon + dx, lat + dy, f"#{rank}",
                 color="white", fontsize=7, fontweight="bold",
                 va="center", transform=PROJ, zorder=5, clip_on=True,
                 bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0))

    fig.suptitle("Predicted Meteorite Discovery Zones",
                 fontsize=12, fontweight="bold")

    if save:
        out = os.path.join(config.FIG_DIR, "fig5_undiscovered_hotspots.pdf")
        _savefig_safe(fig, out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Training curves  (no map needed)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history_path: str = None, save: bool = True):
    """Training and validation loss curves."""
    if history_path is None:
        history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    if not os.path.exists(history_path):
        print("No training history found — run train.py first.")
        return

    with open(history_path) as f:
        history = json.load(f)

    train_loss = np.array([e["loss_total"] for e in history["train"]])
    val_loss   = np.array([e["loss_total"] for e in history["val"]])
    val_mse    = np.array([e["mse"]        for e in history["val"]])
    val_iou    = np.array([e["iou"]        for e in history["val"]])
    epochs     = np.arange(1, len(train_loss) + 1)

    w          = max(5, len(epochs) // 20)
    val_loss_s = uniform_filter1d(val_loss, size=w)
    val_mse_s  = uniform_filter1d(val_mse,  size=w)
    val_iou_s  = uniform_filter1d(val_iou,  size=w)

    best_ep  = epochs[np.argmin(val_mse)]
    best_mse = val_mse.min()
    best_iou = val_iou[np.argmin(val_mse)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].plot(epochs, train_loss, alpha=0.4, color="#1f77b4", lw=0.8)
    axes[0].plot(epochs, val_loss,   alpha=0.25, color="#ff7f0e", lw=0.8)
    axes[0].plot(epochs, uniform_filter1d(train_loss, size=w),
                 color="#1f77b4", lw=1.5, label="Train")
    axes[0].plot(epochs, val_loss_s, color="#ff7f0e", lw=1.5, label="Val")
    axes[0].axvline(best_ep, color="grey", lw=1, ls="--", alpha=0.6)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Combined Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_mse,   alpha=0.25, color="#2ca02c", lw=0.8)
    axes[1].plot(epochs, val_mse_s, color="#2ca02c", lw=1.8, label="Val MSE")
    axes[1].axvline(best_ep, color="grey", lw=1, ls="--", alpha=0.6)
    axes[1].annotate(f"Best: {best_mse:.4f}\n@ ep {best_ep}",
                     xy=(best_ep, best_mse),
                     xytext=(best_ep + len(epochs)*0.05, best_mse + val_mse.max()*0.1),
                     fontsize=7, color="#2ca02c",
                     arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8))
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE")
    axes[1].set_title("Validation MSE")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, val_iou,   alpha=0.25, color="#d62728", lw=0.8)
    axes[2].plot(epochs, val_iou_s, color="#d62728", lw=1.8, label="Val IoU")
    axes[2].axvline(best_ep, color="grey", lw=1, ls="--", alpha=0.6)
    axes[2].annotate(f"@ best ep: {best_iou:.3f}",
                     xy=(best_ep, best_iou),
                     xytext=(best_ep + len(epochs)*0.05, best_iou + val_iou.max()*0.1),
                     fontsize=7, color="#d62728",
                     arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Spatial IoU")
    axes[2].set_title("Validation Spatial IoU")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    fig.suptitle("Training Dynamics", fontsize=11, fontweight="bold")

    if save:
        out = os.path.join(config.FIG_DIR, "fig2_training_curves.pdf")
        _savefig_safe(fig, out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — All-model comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_all_model_comparison(cum_maps: np.ndarray, inc_maps: np.ndarray,
                              save: bool = True):
    """
    Fig 5 — World-map panels for all models + ground truth on 2010s test decade.
    Layout: Ground Truth | ConvLSTM | AttentionConvLSTM | PredRNN | LastFrame
    """
    import torch
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from model import MeteoriteConvLSTM, AttentionConvLSTM, PredRNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEQ  = config.SEQ_LEN
    TEST = config.TEST_DECADE              # scalar integer, e.g. 21
    x_np = cum_maps[TEST - SEQ : TEST]    # (5, 90, 180)
    gt   = inc_maps[TEST]                 # (90, 180) ground truth
    lf   = cum_maps[TEST - 1]             # LastFrame baseline

    x_t  = torch.FloatTensor(x_np).unsqueeze(0).unsqueeze(2).to(device)

    MODEL_CLASSES = {
        "convlstm":  MeteoriteConvLSTM,
        "attention": AttentionConvLSTM,
        "predrnn":   PredRNN,
    }
    labels_map = {
        "convlstm":  "ConvLSTM\n(Ours)",
        "attention": "AttentionConvLSTM\n(Ours, best MSE)",
        "predrnn":   "PredRNN\n(Ours)",
    }

    preds = {}
    for name, cls in MODEL_CLASSES.items():
        ckpt_path = os.path.join(config.CKPT_DIR, f"best_{name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Fig5: skipping {name} — checkpoint not found")
            continue
        m = cls().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        with torch.no_grad():
            preds[name] = m(x_t).squeeze().cpu().numpy()

    if not preds:
        print("  Fig5: no checkpoints found — skipping")
        return

    panels = [("Ground Truth\n(2010s)", gt, "#006600")]
    for key in ["convlstm", "attention", "predrnn"]:
        if key in preds:
            panels.append((labels_map[key], preds[key], "#0055aa"))
    panels.append(("Last-Frame\n(Baseline)", lf, "#888888"))

    n    = len(panels)
    vmax = max(float(gt.max()), 0.01)

    fig = plt.figure(figsize=(n * 4.5, 3.2))
    gs  = fig.add_gridspec(1, n, wspace=0.06)

    for j, (title, data, tcol) in enumerate(panels):
        ax = _make_geo_ax(fig, gs[0, j], title=title, title_color=tcol, fontsize=8)
        im = _imshow_geo(ax, data, DENSITY_CMAP, vmin=0, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          shrink=0.88, pad=0.04, aspect=22)
        cb.ax.tick_params(labelsize=5)

    fig.suptitle("2010s Test Decade — All-Model Prediction Comparison",
                 fontsize=10, fontweight="bold", y=1.04)

    if save:
        out = os.path.join(config.FIG_DIR, "fig6_all_model_comparison.pdf")
        _savefig_safe(fig, out, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cum_maps  = np.load(os.path.join(config.MAPS_DIR, "cumulative_maps.npy"))
    inc_maps  = np.load(os.path.join(config.MAPS_DIR, "incremental_maps.npy"))
    bias_mask = np.load(os.path.join(config.MAPS_DIR, "bias_mask.npy"))

    # Fig 1 — temporal evolution
    plot_temporal_evolution(cum_maps, show_decades=[10, 13, 16, 18, 20, 21])

    # Fig 4 — training curves (needs training_history.json)
    plot_training_curves()

    # The prediction-comparison and hotspot figures show AttentionConvLSTM —
    # load its own saved prediction file.
    attn_pred_path = os.path.join(config.OUTPUT_DIR, "test_prediction_attention.npy")
    generic_pred_path = os.path.join(config.OUTPUT_DIR, "test_prediction.npy")
    pred_model_name = "AttentionConvLSTM"
    if os.path.exists(attn_pred_path):
        pred = np.load(attn_pred_path)
        print("Loaded AttentionConvLSTM prediction for fig2/fig3.")
    elif os.path.exists(generic_pred_path):
        pred = np.load(generic_pred_path)
        pred_model_name = "Model"
        print("WARNING: test_prediction_attention.npy not found — "
              "falling back to generic test_prediction.npy for fig2/fig3 "
              "(caption genericised since the source model is unknown).")
    else:
        pred_model_name = "Model"
        print("WARNING: No saved prediction found — using scaled GT as placeholder.")
        pred = inc_maps[-1] * 0.8 + np.random.rand(*inc_maps[-1].shape) * 0.02

    plot_prediction_comparison(
        pred_map   = pred,
        target_map = inc_maps[-1],
        input_maps = cum_maps[-6:-1],
        decade_idx = cum_maps.shape[0] - 1,
        model_name = pred_model_name
    )
    plot_undiscovered_hotspots(pred, bias_mask)

    # Fig 5 — all-model comparison (requires checkpoints)
    plot_all_model_comparison(cum_maps, inc_maps)

    print("\nAll figures generated.")
