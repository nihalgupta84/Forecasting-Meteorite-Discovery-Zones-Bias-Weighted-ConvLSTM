"""
supplementary_analysis.py
─────────────────────────
Evaluation-only analyses on the released seed-4 models (no training):

  1. sanity      — re-evaluate the seed-4 checkpoints on the 2010s test decade
                   and compare with the seed-4 entries of multiseed_results.json
  2. thresholds  — Spatial IoU swept over binarisation thresholds, plus
                   threshold-free top-k% hotspot overlap
  3. null_island — metrics recomputed after excluding cells within 10 deg of
                   (0, 0), where ~16% of catalogue records carry placeholder
                   coordinates
  4. catalogue   — record shares inside each bias-mask box and composition of
                   the (0, 0) records
  5. hotspots    — coordinates of the top-5 undiscovered-hotspot cells
                   (same peak-picking rule as visualise.py)

Usage:
    python supplementary_analysis.py
Writes outputs/supplementary/supplementary_analysis.json
"""

import json
import os

import numpy as np
import pandas as pd
import torch

import config
from dataset import MeteoriteDensityDataset
from losses import compute_metrics
from model import MeteoriteConvLSTM, AttentionConvLSTM, PredRNN

OUT_DIR = os.path.join(config.OUTPUT_DIR, "supplementary")
os.makedirs(OUT_DIR, exist_ok=True)

TEST = config.TEST_DECADE
LAT_C = -90 + (np.arange(config.GRID_H) + 0.5) * (180 / config.GRID_H)
LON_C = -180 + (np.arange(config.GRID_W) + 0.5) * (360 / config.GRID_W)
LON_G, LAT_G = np.meshgrid(LON_C, LAT_C)


def load_arrays():
    cum = np.load(os.path.join(config.MAPS_DIR, "cumulative_maps.npy")).astype(np.float32)
    inc = np.load(os.path.join(config.MAPS_DIR, "incremental_maps.npy")).astype(np.float32)
    mask = np.load(os.path.join(config.MAPS_DIR, "bias_mask.npy")).astype(np.float32)
    return cum, inc, mask


def model_predictions(cum, inc):
    """Seed-4 checkpoint predictions for the test decade."""
    ds = MeteoriteDensityDataset(cum, inc, [TEST], seq_len=config.SEQ_LEN, augment=False)
    x, y = ds[0]
    x = x.unsqueeze(0)
    preds = {}
    for name, cls in [("convlstm", MeteoriteConvLSTM),
                      ("attention", AttentionConvLSTM),
                      ("predrnn", PredRNN)]:
        m = cls()
        ckpt = torch.load(os.path.join(config.CKPT_DIR, f"best_{name}.pt"), map_location="cpu")
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        with torch.no_grad():
            preds[name] = m(x)[0, 0].numpy()
    return preds, y[0].numpy()


def metrics_np(pred, target, keep=None, tau=config.IOU_THRESHOLD):
    """Same formulas as losses.compute_metrics, optionally restricted to `keep` cells."""
    if keep is None:
        p = torch.tensor(pred[None, None]); t = torch.tensor(target[None, None])
        return compute_metrics(p, t, threshold=tau)
    p, t = pred[keep].astype(np.float64), target[keep].astype(np.float64)
    pb, tb = p > tau, t > tau
    iou = ((pb & tb).sum() + 1e-6) / ((pb | tb).sum() + 1e-6)
    mx, my = p.mean(), t.mean()
    vx, vy = ((p - mx) ** 2).mean(), ((t - my) ** 2).mean()
    cov = ((p - mx) * (t - my)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = (2 * mx * my + c1) * (2 * cov + c2) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2) + 1e-8)
    return {"mse": float(((p - t) ** 2).mean()), "mae": float(np.abs(p - t).mean()),
            "iou": float(iou), "ssim": float(ssim)}


def topk_overlap(pred, target, frac):
    """IoU between the top-`frac` cells of prediction and target (threshold-free)."""
    k = max(int(round(frac * pred.size)), 1)
    pi = set(np.argsort(pred.ravel())[-k:]); ti = set(np.argsort(target.ravel())[-k:])
    return len(pi & ti) / len(pi | ti)


def catalogue_stats():
    import data_preprocessing as dp
    df = dp.load_and_clean()
    n = len(df)
    null = (df["lat"] == 0) & (df["lon"] == 0)
    out = {"n_records": int(n), "null_island": int(null.sum()),
           "null_island_share": float(null.mean())}
    out["null_island_by_decade"] = {int(k): int(v) for k, v in
                                    df[null]["decade"].value_counts().sort_index().items()}
    out["records_by_decade_recent"] = {int(k): int(v) for k, v in
                                       df["decade"].value_counts().sort_index().items() if k >= 1980}
    names = df[null]["name"].str.extract(r"^([A-Za-z' ]+?)\s*\d")[0].str.strip()
    out["null_island_top_name_prefixes"] = names.value_counts().head(8).to_dict()
    out["null_island_found_share"] = float((df[null]["fall"] == "Found").mean())
    boxes = {f"box{i}_{z[0]}_{z[1]}N_{z[2]}_{z[3]}E_w{z[4]}": z for i, z in enumerate(config.BIAS_ZONES)}
    boxes["oman_reference_16_27N_51_60E"] = (16, 27, 51, 60, None)
    out["box_shares"] = {}
    for k, (a, b, c, d, _) in boxes.items():
        m = df["lat"].between(a, b) & df["lon"].between(c, d)
        out["box_shares"][k] = {"count": int(m.sum()), "share": float(m.mean())}
    out["fall_share"] = df["fall"].value_counts(normalize=True).to_dict()
    return out


def hotspots(pred, mask, n=5, suppress_r=5):
    u = (pred * mask).copy()
    pts = []
    for rank in range(n):
        r, c = np.unravel_index(np.argmax(u), u.shape)
        pts.append({"rank": rank + 1, "lat": float(LAT_C[r]), "lon": float(LON_C[c]),
                    "score": float(u[r, c]), "mask_weight": float(mask[r, c])})
        u[max(r - suppress_r, 0):r + suppress_r, max(c - suppress_r, 0):c + suppress_r] = 0
    return pts


def main():
    cum, inc, mask = load_arrays()
    target = inc[TEST]
    preds, target_ds = model_predictions(cum, inc)
    assert np.allclose(target, target_ds)

    # ── 1. sanity ────────────────────────────────────────────────────────────
    ms = json.load(open(os.path.join(config.OUTPUT_DIR, "multiseed_results.json")))
    i4 = ms["seeds"].index(4)
    sanity = {}
    for name, p in preds.items():
        got = metrics_np(p, target)
        ref = {k: ms["main_comparison"][name][k]["per_seed"][i4] for k in ("mse", "mae", "iou", "ssim")}
        saved = np.load(os.path.join(config.OUTPUT_DIR, f"test_prediction_{name}.npy"))
        sanity[name] = {"recomputed": got, "json_seed4": ref,
                        "max_abs_diff": max(abs(got[k] - ref[k]) for k in ref),
                        "saved_npy_max_abs_diff": float(np.abs(saved - p).max())}

    clim = inc[:config.TRAIN_END_DECADE + 1].mean(axis=0)
    static = {"LastFrame": cum[TEST - 1], "ClimatologicalMean": clim / clim.max()}
    lf_ref = ms["main_comparison"]["LastFrame"]
    cl_ref = ms["main_comparison"]["ClimatologicalMean"]
    sanity["ClimatologicalMean"] = {"recomputed": metrics_np(static["ClimatologicalMean"], target),
                                    "json": {k: cl_ref[k]["mean"] for k in ("mse", "mae", "iou", "ssim")}}
    sanity["LastFrame"] = {"recomputed": metrics_np(static["LastFrame"], target),
                           "json": {k: lf_ref[k]["mean"] for k in ("mse", "mae", "iou", "ssim")}}
    # Analyses below use the saved GPU predictions from the reported run
    # (bit-identical to the numbers in multiseed_results.json).
    saved = {n: np.load(os.path.join(config.OUTPUT_DIR, f"test_prediction_{n}.npy")) for n in preds}
    for n, p in saved.items():
        sanity[n]["saved_npy_metrics"] = metrics_np(p, target)
    allp = {**saved, **static}

    # ── 2. thresholds / top-k ────────────────────────────────────────────────
    taus = [round(t, 2) for t in np.arange(0.05, 0.501, 0.05)]
    thr = {name: {"iou_by_tau": {str(t): metrics_np(p, target, tau=t)["iou"] for t in taus},
                  "mean_iou_over_tau": float(np.mean([metrics_np(p, target, tau=t)["iou"] for t in taus])),
                  "top1pct_overlap": topk_overlap(p, target, 0.01),
                  "top5pct_overlap": topk_overlap(p, target, 0.05)}
           for name, p in allp.items()}

    # ── 3. null-island exclusion ─────────────────────────────────────────────
    dist = np.sqrt(LAT_G ** 2 + LON_G ** 2)
    keep = dist > 10.0
    tgt_peak = np.unravel_index(np.argmax(target), target.shape)
    lf_peak = np.unravel_index(np.argmax(inc[TEST - 1]), target.shape)
    null = {"radius_deg": 10.0, "excluded_cells": int((~keep).sum()),
            "target_peak_latlon": [float(LAT_C[tgt_peak[0]]), float(LON_C[tgt_peak[1]])],
            "inc2000s_peak_latlon": [float(LAT_C[lf_peak[0]]), float(LON_C[lf_peak[1]])],
            "target_cells_above_tau_total": int((target > 0.1).sum()),
            "target_cells_above_tau_in_excluded": int(((target > 0.1) & ~keep).sum()),
            "target_mass_share_in_excluded": float(target[~keep].sum() / target.sum()),
            "metrics_full": {n: metrics_np(p, target) for n, p in allp.items()},
            "metrics_excluded": {n: metrics_np(p, target, keep=keep) for n, p in allp.items()}}

    out = {"note": "seed-4 checkpoints / deterministic baselines; single seed",
           "sanity": sanity, "thresholds": thr, "null_island": null,
           "catalogue": catalogue_stats(),
           "hotspots_attention": hotspots(saved["attention"], mask)}
    path = os.path.join(OUT_DIR, "supplementary_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
