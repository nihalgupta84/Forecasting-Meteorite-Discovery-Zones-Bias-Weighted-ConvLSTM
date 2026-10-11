"""
kde_maps.py
───────────
Generates KDE (Kernel Density Estimation) density maps from lat/lon coordinates.
Produces:
  - Incremental maps  (only new landings in that decade)
  - Cumulative maps   (all landings up to that decade)
  - Bias weight mask  (down-weights collection program zones)

Each map is a 2D numpy array of shape (GRID_H, GRID_W) normalised to [0, 1].

Run standalone:  python kde_maps.py
"""

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import config
import data_preprocessing as dp


# ── Grid definition ────────────────────────────────────────────────────────────

# Latitude  axis: from -90 to +90  in GRID_H steps  (cell centres)
LAT_BINS = np.linspace(-90  + 90 / config.GRID_H,
                        90  - 90 / config.GRID_H,
                        config.GRID_H)

# Longitude axis: from -180 to +180 in GRID_W steps (cell centres)
LON_BINS = np.linspace(-180 + 180 / config.GRID_W,
                        180 - 180 / config.GRID_W,
                        config.GRID_W)

# Meshgrid for KDE evaluation: shape (GRID_H * GRID_W, 2)
LON_GRID, LAT_GRID = np.meshgrid(LON_BINS, LAT_BINS)  # both (H, W)
EVAL_POINTS = np.vstack([LAT_GRID.ravel(), LON_GRID.ravel()])  # (2, H*W)


def coords_to_kde_map(coords: np.ndarray,
                      bandwidth: float = config.KDE_BANDWIDTH,
                      min_points: int = 3) -> np.ndarray:
    """
    Given N×2 lat/lon coordinates, return a KDE density map
    of shape (GRID_H, GRID_W) normalised to [0, 1].

    If fewer than min_points records, returns a zero map.
    """
    if len(coords) < min_points:
        return np.zeros((config.GRID_H, config.GRID_W), dtype=np.float32)

    # scipy gaussian_kde expects shape (2, N): [lat, lon]
    data = coords.T.astype(np.float64)   # (2, N)

    # Use Silverman's rule for bandwidth if bandwidth <= 0
    if bandwidth <= 0:
        kde = gaussian_kde(data, bw_method="silverman")
    else:
        # Convert degree bandwidth to fraction of data range (scipy convention)
        # bw_method = scalar → std factor; we pass the covariance factor
        n = data.shape[1]
        std_lat = data[0].std() + 1e-6
        std_lon = data[1].std() + 1e-6
        # Normalise data, set bandwidth, un-normalise conceptually
        kde = gaussian_kde(data, bw_method=bandwidth / max(std_lat, std_lon))

    density = kde(EVAL_POINTS)           # shape (H*W,)
    density = density.reshape(config.GRID_H, config.GRID_W).astype(np.float32)

    # Normalise to [0, 1]
    d_max = density.max()
    if d_max > 0:
        density /= d_max

    return density


def build_bias_mask(zones=None) -> np.ndarray:
    """
    Returns a weight map of shape (GRID_H, GRID_W) where:
      - Known collection-program zones get lower weight (e.g. 0.3)
      - Unbiased zones get weight 1.0

    Used to compute the bias-masked MSE loss during training.

    Args:
        zones: list of (lat_min, lat_max, lon_min, lon_max, weight) tuples.
               Defaults to config.BIAS_ZONES; pass an alternate list to build
               a mask for the bias-mask-weight sensitivity ablation without
               mutating global config.
    """
    zones = zones if zones is not None else config.BIAS_ZONES
    mask = np.ones((config.GRID_H, config.GRID_W), dtype=np.float32)

    for (lat_min, lat_max, lon_min, lon_max, weight) in zones:
        lat_idx = np.where((LAT_BINS >= lat_min) & (LAT_BINS <= lat_max))[0]
        lon_idx = np.where((LON_BINS >= lon_min) & (LON_BINS <= lon_max))[0]
        if len(lat_idx) > 0 and len(lon_idx) > 0:
            row_start, row_end = lat_idx[0],  lat_idx[-1] + 1
            col_start, col_end = lon_idx[0],  lon_idx[-1] + 1
            mask[row_start:row_end, col_start:col_end] = weight

    return mask


def generate_all_maps(df,
                      save_npy: bool = True,
                      bandwidth: float = None) -> dict:
    """
    Generate incremental and cumulative KDE maps for every decade.

    Args:
        bandwidth: KDE bandwidth override in degrees (defaults to
                   config.KDE_BANDWIDTH). Used by the bandwidth sensitivity
                   ablation to regenerate maps without touching the on-disk
                   default maps (pass save_npy=False in that case).

    Returns:
        {
          "incremental": {decade_idx: np.ndarray (H, W)},
          "cumulative":  {decade_idx: np.ndarray (H, W)},
          "bias_mask":   np.ndarray (H, W),
          "num_decades": int
        }
    """
    bandwidth = bandwidth if bandwidth is not None else config.KDE_BANDWIDTH
    incremental_coords = dp.get_decade_coords(df)
    cumulative_coords  = dp.get_cumulative_coords(df)

    num_decades = max(incremental_coords.keys()) + 1
    print(f"Generating KDE maps for {num_decades} decades (bandwidth={bandwidth})…")

    incremental_maps = {}
    cumulative_maps  = {}

    for idx in range(num_decades):
        year = dp.decade_idx_to_year(idx)
        print(f"  Decade {idx:02d} ({year}s)  — "
              f"incremental: {len(incremental_coords[idx]):4d} pts  |  "
              f"cumulative: {len(cumulative_coords[idx]):5d} pts")

        inc_map = coords_to_kde_map(incremental_coords[idx], bandwidth=bandwidth)
        cum_map = coords_to_kde_map(cumulative_coords[idx], bandwidth=bandwidth)

        incremental_maps[idx] = inc_map
        cumulative_maps[idx]  = cum_map

    bias_mask = build_bias_mask()

    result = {
        "incremental": incremental_maps,
        "cumulative":  cumulative_maps,
        "bias_mask":   bias_mask,
        "num_decades": num_decades,
    }

    if save_npy:
        # Stack all maps into arrays for easy loading
        inc_stack = np.stack(
            [incremental_maps[i] for i in range(num_decades)], axis=0
        )   # (num_decades, H, W)
        cum_stack = np.stack(
            [cumulative_maps[i] for i in range(num_decades)], axis=0
        )   # (num_decades, H, W)

        np.save(os.path.join(config.MAPS_DIR, "incremental_maps.npy"), inc_stack)
        np.save(os.path.join(config.MAPS_DIR, "cumulative_maps.npy"),  cum_stack)
        np.save(os.path.join(config.MAPS_DIR, "bias_mask.npy"),        bias_mask)
        print(f"\nSaved .npy files to: {config.MAPS_DIR}")

    return result


def visualise_maps(maps_dict: dict, n_cols: int = 4) -> None:
    """
    Save a grid figure showing all cumulative KDE maps across decades.
    """
    cum_maps   = maps_dict["cumulative"]
    num_decades = maps_dict["num_decades"]
    n_rows = (num_decades + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 2.5))
    axes = axes.flatten()

    for idx in range(num_decades):
        ax  = axes[idx]
        img = cum_maps[idx]
        year = dp.decade_idx_to_year(idx)
        im = ax.imshow(img, origin="lower", cmap="hot",
                       aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{year}s", fontsize=8)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(num_decades, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Cumulative Meteorite Landing Density Maps (KDE)", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(config.FIG_DIR, "all_cumulative_maps.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved map grid: {out_path}")


def visualise_bias_mask(mask: np.ndarray) -> None:
    """Save a figure of the discovery bias weight mask."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(mask, origin="lower", cmap="RdYlGn",
                   aspect="auto", vmin=0, vmax=1,
                   extent=[-180, 180, -90, 90])
    plt.colorbar(im, ax=ax, label="Loss weight")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Discovery Bias Mask\n"
                 "(green = unbiased, red = collection-program-dominated)")

    # Add zone labels
    zone_labels = [
        (-80, 0,   "Antarctica\n(ANSMET)"),
        ( 25, 30,  "Sahara /\nArabia"),
        ( 30, 100, "Oman /\nPakistan"),
    ]
    for lat, lon, label in zone_labels:
        ax.text(lon, lat, label, ha="center", va="center",
                fontsize=7, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    plt.tight_layout()
    out_path = os.path.join(config.FIG_DIR, "bias_mask.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved bias mask: {out_path}")


if __name__ == "__main__":
    df       = dp.load_and_clean()
    maps     = generate_all_maps(df, save_npy=True)
    visualise_maps(maps)
    visualise_bias_mask(maps["bias_mask"])
    print("\nKDE map generation complete.")
