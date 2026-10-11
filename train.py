"""
train.py
────────
Training and evaluation pipeline for MeteoriteConvLSTM and all experiments.

Modes
─────
  python train.py                         # train ConvLSTM (default, 200 epochs)
  python train.py --model attention       # train AttentionConvLSTM
  python train.py --model predrnn         # train PredRNN
  python train.py --all-models            # train ConvLSTM + Attention + PredRNN
  python train.py --rolling-cv            # rolling cross-validation (4 test decades)
  python train.py --ablate depth          # ablation: encoder depth
  python train.py --ablate seqlen         # ablation: sequence length
  python train.py --ablate warmup         # ablation: loss warmup vs no warmup
  python train.py --ablate all            # run all ablations
  python train.py --full-pipeline         # everything: all models + CV + ablations
  python train.py --no-wandb              # disable Weights & Biases logging
"""

import argparse
import os
import time
import json
import copy
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

# Load optional .env (e.g. WANDB_API_KEY) from the repository root if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import config
import data_preprocessing as dp
import kde_maps
from dataset import build_dataloaders, build_dataloaders_for_cv, load_maps_from_disk
from model import (MeteoriteConvLSTM, AttentionConvLSTM, PredRNN,
                   LastFrameBaseline, SingleFrameCNN, LSTMFlatBaseline,
                   count_parameters)
from losses import CombinedLoss, compute_metrics


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int):
    """
    Reset every RNG and force deterministic CUDA kernels.

    Must be called immediately before instantiating EACH independently-reported
    model (main comparison, every rolling-CV fold, every ablation variant).
    Without this, sequential trainings within one process share a single
    drifting RNG stream, so results depend on execution order and are not
    reproducible in isolation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(config.SEED)

WANDB_PROJECT = "meteorite-prediction"

# ── W&B helpers ────────────────────────────────────────────────────────────────

def wandb_active():
    """True when a wandb run is live and logging is enabled."""
    return wandb.run is not None


def _log(metrics: dict, step: int = None):
    """Log to wandb only when a run is active."""
    if wandb_active():
        wandb.log(metrics, step=step)


# ══════════════════════════════════════════════════════════════════════════════
# Core training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    all_comps = {k: 0.0 for k in
                 ["loss_total", "loss_mse", "loss_iou", "loss_bias"]}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss, comps = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for k, v in comps.items():
            all_comps[k] += v
    n = len(loader)
    return {k: v / n for k, v in all_comps.items()}


def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_metrics = {k: 0.0 for k in
                   ["loss_total", "loss_mse", "loss_iou", "loss_bias",
                    "mse", "mae", "iou", "ssim"]}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, comps = loss_fn(pred, y)
            metrics  = compute_metrics(pred, y)
            for k, v in {**comps, **metrics}.items():
                all_metrics[k] += v
    n = max(len(loader), 1)
    return {k: v / n for k, v in all_metrics.items()}


def train_model(model, train_dl, val_dl, epochs, device, bias_mask,
                lr=None, use_warmup=True, tag="model", verbose=True,
                wandb_prefix: str = ""):
    """
    Generic training loop for any model with the same interface.
    Returns best val_metrics and training history.

    wandb_prefix: metric key prefix, e.g. "convlstm/" so metrics appear as
                  "convlstm/train/loss_total" in the dashboard.
    """
    lr = lr or config.LR
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    loss_fn       = CombinedLoss(bias_mask).to(device)
    mse_only_loss = CombinedLoss(bias_mask,
                                  lambda_mse=1.0,
                                  lambda_iou=0.0,
                                  lambda_bias=0.5).to(device)

    # Dual checkpoint: one for best MSE, one for best composite score.
    # Composite = val_mse - CKPT_IOU_WEIGHT * val_iou (lower = better).
    # We only start tracking composite AFTER warmup ends, so the model has
    # had time to learn non-trivial predictions.
    CKPT_IOU_WEIGHT  = 0.05   # small — keeps MSE primary, breaks ties with IoU

    best_val_mse     = float("inf")
    best_composite   = float("inf")
    best_mse_state   = None
    best_iou_state   = None   # checkpoint that maximises val_iou post-warmup
    best_val_iou     = -1.0
    history          = {"train": [], "val": []}

    warmup_ep = config.LOSS_WARMUP_EPOCHS if use_warmup else 0
    if verbose:
        print(f"\n  [{tag}] Training {epochs} ep "
              f"({'warmup ' + str(warmup_ep) + ' ep, ' if use_warmup else ''}"
              f"params={count_parameters(model):,})")

    pfx = wandb_prefix  # e.g. "convlstm/"

    for epoch in range(1, epochs + 1):
        t0          = time.time()
        active_loss = mse_only_loss if epoch <= warmup_ep else loss_fn
        tr          = train_one_epoch(model, train_dl, optimizer, active_loss, device)
        va          = evaluate(model, val_dl, loss_fn, device)
        scheduler.step()
        current_lr  = scheduler.get_last_lr()[0]

        history["train"].append(tr)
        history["val"].append(va)

        # --- Checkpoint 1: best val_mse (original) ---
        if va["mse"] < best_val_mse:
            best_val_mse  = va["mse"]
            best_mse_state = copy.deepcopy(model.state_dict())

        # --- Checkpoint 2: best val_iou post-warmup (new) ---
        if epoch > warmup_ep and va["iou"] > best_val_iou:
            best_val_iou  = va["iou"]
            best_iou_state = copy.deepcopy(model.state_dict())

        # --- Checkpoint 3: best composite score post-warmup ---
        composite = va["mse"] - CKPT_IOU_WEIGHT * va["iou"]
        if epoch > warmup_ep and composite < best_composite:
            best_composite = composite
            best_state     = copy.deepcopy(model.state_dict())
        elif epoch <= warmup_ep:
            # During warmup fall back to mse-only composite tracking
            best_state = best_mse_state

        # ── W&B per-epoch logging ──────────────────────────────────────────────
        _log({
            f"{pfx}train/loss_total":  tr["loss_total"],
            f"{pfx}train/loss_mse":    tr["loss_mse"],
            f"{pfx}train/loss_iou":    tr["loss_iou"],
            f"{pfx}train/loss_bias":   tr["loss_bias"],
            f"{pfx}val/loss_total":    va["loss_total"],
            f"{pfx}val/mse":           va["mse"],
            f"{pfx}val/mae":           va["mae"],
            f"{pfx}val/iou":           va["iou"],
            f"{pfx}val/ssim":          va["ssim"],
            f"{pfx}val/best_mse":      best_val_mse,
            f"{pfx}lr":                current_lr,
            f"{pfx}epoch":             epoch,
            f"{pfx}warmup_active":     int(epoch <= warmup_ep),
            f"{pfx}epoch_time_s":      time.time() - t0,
        })
        # ──────────────────────────────────────────────────────────────────────

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"    ep {epoch:3d}/{epochs}  "
                  f"tr_loss={tr['loss_total']:.4f}  "
                  f"val_mse={va['mse']:.4f}  "
                  f"val_iou={va['iou']:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    # Restore best composite-score weights (balances MSE + IoU post-warmup)
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(best_mse_state)
    return history, best_val_mse


def train_baseline_model(model, train_dl, device, bias_mask,
                          epochs=None, tag="baseline"):
    """
    Train a learned baseline with cosine-annealing LR.
    Uses same epochs as main models for fair comparison.
    """
    epochs  = epochs or config.BASELINE_TRAIN_EPOCHS
    loss_fn = CombinedLoss(bias_mask).to(device)
    opt     = optim.Adam(model.parameters(), lr=config.LR,
                         weight_decay=config.WEIGHT_DECAY)
    sched   = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_val = float("inf")
    best_state = None
    print(f"  [{tag}] Training {epochs} ep")
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss_val, _ = loss_fn(pred, y)
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss_val.item()
        sched.step()
        avg_loss = total_loss / max(len(train_dl), 1)
        if avg_loss < best_val:
            best_val   = avg_loss
            best_state = copy.deepcopy(model.state_dict())
        _log({f"baseline/{tag}/train_loss": avg_loss,
              f"baseline/{tag}/epoch": ep + 1})
    if best_state:
        model.load_state_dict(best_state)


# ══════════════════════════════════════════════════════════════════════════════
# Climatological Mean Baseline
# ══════════════════════════════════════════════════════════════════════════════

def climatological_mean_baseline(inc_maps, train_end_decade):
    """
    Predict the mean of all training incremental maps.
    This is a strong naive baseline for geospatial data — it captures
    the average spatial distribution without any temporal modelling.

    Returns:
        mean_map : (H, W) numpy array, normalised to [0,1]
    """
    train_maps = inc_maps[:train_end_decade + 1]   # up to (including) train_end
    mean_map   = train_maps.mean(axis=0)
    if mean_map.max() > 0:
        mean_map = mean_map / mean_map.max()
    return mean_map


def eval_static_prediction(pred_map, target_map):
    """
    Evaluate a static numpy prediction (no model, no loader).
    Returns dict with mse, mae, iou, ssim.
    """
    pred_t   = torch.tensor(pred_map[None, None],  dtype=torch.float32)
    target_t = torch.tensor(target_map[None, None], dtype=torch.float32)
    return compute_metrics(pred_t, target_t)


# ══════════════════════════════════════════════════════════════════════════════
# Rolling Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════

def rolling_cv(model_class, model_kwargs, cum_maps, inc_maps, bias_mask,
               test_decades, epochs, device, tag="ConvLSTM", seed=None):
    """
    Rolling cross-validation: for each test_decade d, train on decades
    up to d-1 and evaluate on d.

    Returns:
        cv_results : list of dicts (one per test decade)
        cv_summary : dict with mean ± std of each metric
    """
    print(f"\n{'─'*55}")
    print(f"  Rolling CV — {tag}  ({len(test_decades)} folds)")
    print(f"{'─'*55}")

    loss_fn = CombinedLoss(bias_mask).to(device)
    cv_results = []

    for test_d in test_decades:
        val_d    = test_d - 1     # validation = one decade before test
        train_end = val_d - 1

        if train_end < config.SEQ_LEN:
            print(f"  Skipping test decade {test_d} — not enough history.")
            continue

        print(f"\n  Fold: test={test_d} ({1800+test_d*10}s)  "
              f"val={val_d}  train=0..{train_end}")

        # Build dataloaders for this fold
        train_dl, val_dl, test_dl = build_dataloaders_for_cv(
            cum_maps, inc_maps,
            train_end_decade=train_end,
            val_decade=val_d,
            test_decade=test_d,
            seq_len=config.SEQ_LEN,
            batch_size=config.BATCH_SIZE,
        )

        # Fresh model for each fold — reseed so each fold is independently
        # reproducible regardless of what trained before it in this process.
        set_seed(seed if seed is not None else config.SEED)
        model = model_class(**model_kwargs).to(device)
        fold_tag = f"{tag}/test={test_d}"
        pfx      = f"cv/{tag}/fold{test_d}/"

        train_model(model, train_dl, val_dl, epochs, device, bias_mask,
                    tag=fold_tag, verbose=True, wandb_prefix=pfx)

        test_metrics = evaluate(model, test_dl, loss_fn, device)
        test_metrics["test_decade"] = test_d
        test_metrics["test_year"]   = 1800 + test_d * 10
        cv_results.append(test_metrics)

        print(f"    → MSE={test_metrics['mse']:.4f}  "
              f"IoU={test_metrics['iou']:.4f}  "
              f"SSIM={test_metrics['ssim']:.4f}")

        # Log fold test metrics
        _log({
            f"cv/{tag}/fold_mse":  test_metrics["mse"],
            f"cv/{tag}/fold_mae":  test_metrics["mae"],
            f"cv/{tag}/fold_iou":  test_metrics["iou"],
            f"cv/{tag}/fold_ssim": test_metrics["ssim"],
            f"cv/{tag}/test_year": test_metrics["test_year"],
        })

    # Summary statistics
    if cv_results:
        cv_summary = {}
        for metric in ["mse", "mae", "iou", "ssim"]:
            vals = [r[metric] for r in cv_results]
            cv_summary[metric + "_mean"] = float(np.mean(vals))
            cv_summary[metric + "_std"]  = float(np.std(vals))
        print(f"\n  CV Summary ({tag}):")
        print(f"    MSE  {cv_summary['mse_mean']:.4f} ± {cv_summary['mse_std']:.4f}")
        print(f"    IoU  {cv_summary['iou_mean']:.4f} ± {cv_summary['iou_std']:.4f}")
        print(f"    SSIM {cv_summary['ssim_mean']:.4f} ± {cv_summary['ssim_std']:.4f}")

        # Log CV summary to wandb
        _log({
            f"cv/{tag}/summary/mse_mean":  cv_summary["mse_mean"],
            f"cv/{tag}/summary/mse_std":   cv_summary["mse_std"],
            f"cv/{tag}/summary/iou_mean":  cv_summary["iou_mean"],
            f"cv/{tag}/summary/iou_std":   cv_summary["iou_std"],
            f"cv/{tag}/summary/ssim_mean": cv_summary["ssim_mean"],
            f"cv/{tag}/summary/ssim_std":  cv_summary["ssim_std"],
        })

        # Log CV fold results as a wandb Table
        if wandb_active():
            table = wandb.Table(columns=["model", "test_decade", "test_year",
                                          "mse", "mae", "iou", "ssim"])
            for r in cv_results:
                table.add_data(tag, r["test_decade"], r["test_year"],
                               r["mse"], r["mae"], r["iou"], r["ssim"])
            wandb.log({f"cv/{tag}/fold_table": table})
    else:
        cv_summary = {}

    return cv_results, cv_summary


# ══════════════════════════════════════════════════════════════════════════════
# Ablation Studies
# ══════════════════════════════════════════════════════════════════════════════

def ablation_depth(cum_maps, inc_maps, bias_mask, train_dl, val_dl, test_dl,
                   epochs, device, seed=None):
    """
    Ablation: encoder depth — 1, 2, 3 ConvLSTM layers.
    Uses a modified MeteoriteConvLSTM with variable depth.
    """
    print(f"\n{'─'*55}")
    print(f"  Ablation: Encoder Depth")
    print(f"{'─'*55}")
    loss_fn = CombinedLoss(bias_mask).to(device)
    results = {}
    rows    = []

    for hidden_list in config.ABLATION_DEPTHS:
        tag = f"depth{len(hidden_list)}_ch{'x'.join(map(str,hidden_list))}"
        set_seed(seed if seed is not None else config.SEED)
        model = MeteoriteConvLSTM(hidden_channels=hidden_list).to(device)
        train_model(model, train_dl, val_dl, epochs, device, bias_mask,
                    tag=tag, verbose=True, wandb_prefix=f"ablation/depth/{tag}/")
        m = evaluate(model, test_dl, loss_fn, device)
        results[tag] = {
            "depth": len(hidden_list),
            "channels": hidden_list,
            "params": count_parameters(model),
            "mse": m["mse"], "mae": m["mae"],
            "iou": m["iou"], "ssim": m["ssim"],
        }
        rows.append(results[tag])
        print(f"  {tag}: MSE={m['mse']:.4f}  IoU={m['iou']:.4f}  SSIM={m['ssim']:.4f}")

        _log({
            f"ablation/depth/{tag}/test_mse":  m["mse"],
            f"ablation/depth/{tag}/test_iou":  m["iou"],
            f"ablation/depth/{tag}/test_ssim": m["ssim"],
        })

    if wandb_active() and rows:
        table = wandb.Table(columns=["depth", "channels", "params",
                                      "test_mse", "test_mae", "test_iou", "test_ssim"])
        for r in rows:
            table.add_data(r["depth"], str(r["channels"]), r["params"],
                           r["mse"], r["mae"], r["iou"], r["ssim"])
        wandb.log({"ablation/depth_table": table})

    return results


def ablation_seqlen(cum_maps, inc_maps, bias_mask, device, epochs, seed=None):
    """
    Ablation: input sequence length T ∈ {1, 3, 5, 7}.
    Must rebuild dataloaders for each T.
    """
    print(f"\n{'─'*55}")
    print(f"  Ablation: Sequence Length")
    print(f"{'─'*55}")
    loss_fn = CombinedLoss(bias_mask).to(device)
    results = {}
    rows    = []

    for T in config.ABLATION_SEQ_LENS:
        tag = f"seqlen{T}"
        tr_dl, va_dl, te_dl = build_dataloaders(
            cum_maps, inc_maps, seq_len=T, batch_size=config.BATCH_SIZE)
        set_seed(seed if seed is not None else config.SEED)
        model = MeteoriteConvLSTM().to(device)
        train_model(model, tr_dl, va_dl, epochs, device, bias_mask,
                    tag=tag, verbose=True, wandb_prefix=f"ablation/seqlen/{tag}/")
        m = evaluate(model, te_dl, loss_fn, device)
        results[tag] = {
            "seq_len": T,
            "mse": m["mse"], "mae": m["mae"],
            "iou": m["iou"], "ssim": m["ssim"],
        }
        rows.append({"seq_len": T, **m})
        print(f"  T={T}: MSE={m['mse']:.4f}  IoU={m['iou']:.4f}  SSIM={m['ssim']:.4f}")

        _log({
            f"ablation/seqlen/T{T}/test_mse":  m["mse"],
            f"ablation/seqlen/T{T}/test_iou":  m["iou"],
            f"ablation/seqlen/T{T}/test_ssim": m["ssim"],
        })

    if wandb_active() and rows:
        table = wandb.Table(columns=["seq_len", "test_mse", "test_mae",
                                      "test_iou", "test_ssim"])
        for r in rows:
            table.add_data(r["seq_len"], r["mse"], r["mae"], r["iou"], r["ssim"])
        wandb.log({"ablation/seqlen_table": table})

    return results


def ablation_warmup(train_dl, val_dl, test_dl, bias_mask, device, epochs, seed=None):
    """
    Ablation: loss warmup vs no warmup.
    Proves the warmup prevents all-zero collapse.
    """
    print(f"\n{'─'*55}")
    print(f"  Ablation: Loss Warmup")
    print(f"{'─'*55}")
    loss_fn = CombinedLoss(bias_mask).to(device)
    results = {}

    for use_warmup, tag in [(True, "with_warmup"), (False, "no_warmup")]:
        set_seed(seed if seed is not None else config.SEED)
        model = MeteoriteConvLSTM().to(device)
        train_model(model, train_dl, val_dl, epochs, device, bias_mask,
                    use_warmup=use_warmup, tag=tag, verbose=True,
                    wandb_prefix=f"ablation/warmup/{tag}/")
        m = evaluate(model, test_dl, loss_fn, device)
        results[tag] = {
            "warmup": use_warmup,
            "mse": m["mse"], "mae": m["mae"],
            "iou": m["iou"], "ssim": m["ssim"],
        }
        print(f"  {tag}: MSE={m['mse']:.4f}  IoU={m['iou']:.4f}  SSIM={m['ssim']:.4f}")
        _log({
            f"ablation/warmup/{tag}/test_mse":  m["mse"],
            f"ablation/warmup/{tag}/test_iou":  m["iou"],
            f"ablation/warmup/{tag}/test_ssim": m["ssim"],
        })

    if wandb_active():
        table = wandb.Table(columns=["variant", "use_warmup",
                                      "test_mse", "test_mae", "test_iou", "test_ssim"])
        for tag, r in results.items():
            table.add_data(tag, r["warmup"], r["mse"], r["mae"], r["iou"], r["ssim"])
        wandb.log({"ablation/warmup_table": table})

    return results


def ablation_loss_components(train_dl, val_dl, test_dl, bias_mask, device, epochs, seed=None):
    """
    Ablation: contribution of each loss term.
    Trains 4 variants: full | no-IoU | no-bias | MSE-only
    """
    print(f"\n{'─'*55}")
    print(f"  Ablation: Loss Components")
    print(f"{'─'*55}")
    results = {}

    loss_configs = {
        "full_loss":    dict(lambda_mse=1.0, lambda_iou=0.1, lambda_bias=0.5),
        "no_iou":       dict(lambda_mse=1.0, lambda_iou=0.0, lambda_bias=0.5),
        "no_bias":      dict(lambda_mse=1.0, lambda_iou=0.1, lambda_bias=0.0),
        "mse_only":     dict(lambda_mse=1.0, lambda_iou=0.0, lambda_bias=0.0),
    }

    eval_loss = CombinedLoss(bias_mask).to(device)

    for tag, lc in loss_configs.items():
        set_seed(seed if seed is not None else config.SEED)
        model    = MeteoriteConvLSTM().to(device)
        loss_fn  = CombinedLoss(bias_mask, **lc).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.LR,
                               weight_decay=config.WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        best_val_mse = float("inf")
        best_state   = None

        mse_only = CombinedLoss(bias_mask,
                                lambda_mse=1.0, lambda_iou=0.0, lambda_bias=0.5
                                ).to(device)
        pfx = f"ablation/loss/{tag}/"
        for ep in range(1, epochs + 1):
            active = mse_only if ep <= config.LOSS_WARMUP_EPOCHS else loss_fn
            tr = train_one_epoch(model, train_dl, optimizer, active, device)
            va = evaluate(model, val_dl, eval_loss, device)
            scheduler.step()
            if va["mse"] < best_val_mse:
                best_val_mse = va["mse"]
                best_state   = copy.deepcopy(model.state_dict())
            _log({f"{pfx}train/loss_total": tr["loss_total"],
                  f"{pfx}val/mse": va["mse"],
                  f"{pfx}val/iou": va["iou"],
                  f"{pfx}epoch": ep})

        model.load_state_dict(best_state)
        m = evaluate(model, test_dl, eval_loss, device)
        results[tag] = {
            "config": lc,
            "mse": m["mse"], "mae": m["mae"],
            "iou": m["iou"], "ssim": m["ssim"],
        }
        print(f"  {tag:<15s}: MSE={m['mse']:.4f}  IoU={m['iou']:.4f}  SSIM={m['ssim']:.4f}")
        _log({f"ablation/loss/{tag}/test_mse":  m["mse"],
              f"ablation/loss/{tag}/test_iou":  m["iou"],
              f"ablation/loss/{tag}/test_ssim": m["ssim"]})

    if wandb_active():
        table = wandb.Table(columns=["loss_config", "lambda_mse", "lambda_iou",
                                      "lambda_bias", "test_mse", "test_mae",
                                      "test_iou", "test_ssim"])
        for tag, r in results.items():
            table.add_data(tag,
                           r["config"]["lambda_mse"],
                           r["config"]["lambda_iou"],
                           r["config"]["lambda_bias"],
                           r["mse"], r["mae"], r["iou"], r["ssim"])
        wandb.log({"ablation/loss_components_table": table})

    return results


def ablation_bandwidth(df, bias_mask, epochs, device, seed=None):
    """
    Ablation: KDE bandwidth h ∈ config.ABLATION_KDE_BWS (degrees).
    Regenerates KDE maps at each bandwidth (in-memory only — never
    overwrites the on-disk default maps) and retrains the primary
    ConvLSTM config, holding architecture/loss/seed fixed so the only
    varying factor is the bandwidth used to build the input density maps.
    """
    print(f"\n{'─'*55}")
    print(f"  Ablation: KDE Bandwidth")
    print(f"{'─'*55}")
    loss_fn = CombinedLoss(bias_mask).to(device)
    results = {}

    for bw in config.ABLATION_KDE_BWS:
        tag = f"bw{bw:g}"
        maps = kde_maps.generate_all_maps(df, save_npy=False, bandwidth=bw)
        cum_maps = np.stack([maps["cumulative"][i]
                              for i in range(maps["num_decades"])], axis=0)
        inc_maps = np.stack([maps["incremental"][i]
                              for i in range(maps["num_decades"])], axis=0)
        tr_dl, va_dl, te_dl = build_dataloaders(
            cum_maps, inc_maps, seq_len=config.SEQ_LEN, batch_size=config.BATCH_SIZE)

        set_seed(seed if seed is not None else config.SEED)
        model = MeteoriteConvLSTM().to(device)
        train_model(model, tr_dl, va_dl, epochs, device, bias_mask,
                    tag=tag, verbose=True, wandb_prefix=f"ablation/bandwidth/{tag}/")
        m = evaluate(model, te_dl, loss_fn, device)
        results[tag] = {
            "bandwidth": bw,
            "mse": m["mse"], "mae": m["mae"],
            "iou": m["iou"], "ssim": m["ssim"],
        }
        print(f"  h={bw}: MSE={m['mse']:.4f}  IoU={m['iou']:.4f}  SSIM={m['ssim']:.4f}")
        _log({
            f"ablation/bandwidth/{tag}/test_mse":  m["mse"],
            f"ablation/bandwidth/{tag}/test_iou":  m["iou"],
            f"ablation/bandwidth/{tag}/test_ssim": m["ssim"],
        })

    if wandb_active():
        table = wandb.Table(columns=["bandwidth", "test_mse", "test_mae",
                                      "test_iou", "test_ssim"])
        for r in results.values():
            table.add_data(r["bandwidth"], r["mse"], r["mae"], r["iou"], r["ssim"])
        wandb.log({"ablation/bandwidth_table": table})

    return results


def ablation_bias_mask(train_dl, val_dl, test_dl, epochs, device, seed=None):
    """
    Ablation: bias-mask weight sensitivity.
    Sweeps config.ABLATION_BIAS_WEIGHTS (antarctica_weight, sahara_weight)
    through a freshly-built mask, holding architecture/seq-len/seed fixed
    so the only varying factor is the mask's down-weighting strength.
    """
    print(f"\n{'─'*55}")
    print(f"  Ablation: Bias-Mask Weight Sensitivity")
    print(f"{'─'*55}")
    results = {}

    for tag, (w_antarctica, w_sahara) in config.ABLATION_BIAS_WEIGHTS.items():
        zones = [
            (-90, -70, -180, 180, w_antarctica),   # Antarctica
            ( 15,  35,    5,  55, w_sahara),       # Sahara / Arabian Peninsula
            ( 20,  40,   80, 120, w_sahara),       # Oman / Pakistan hot desert
        ]
        mask = kde_maps.build_bias_mask(zones=zones)
        loss_fn = CombinedLoss(mask).to(device)

        set_seed(seed if seed is not None else config.SEED)
        model = MeteoriteConvLSTM().to(device)
        train_model(model, train_dl, val_dl, epochs, device, mask,
                    tag=tag, verbose=True, wandb_prefix=f"ablation/bias_mask/{tag}/")
        m = evaluate(model, test_dl, loss_fn, device)
        results[tag] = {
            "antarctica_weight": w_antarctica,
            "sahara_weight": w_sahara,
            "mse": m["mse"], "mae": m["mae"],
            "iou": m["iou"], "ssim": m["ssim"],
        }
        print(f"  {tag} (w_ant={w_antarctica}, w_sah={w_sahara}): "
              f"MSE={m['mse']:.4f}  IoU={m['iou']:.4f}  SSIM={m['ssim']:.4f}")
        _log({
            f"ablation/bias_mask/{tag}/test_mse":  m["mse"],
            f"ablation/bias_mask/{tag}/test_iou":  m["iou"],
            f"ablation/bias_mask/{tag}/test_ssim": m["ssim"],
        })

    if wandb_active():
        table = wandb.Table(columns=["variant", "w_antarctica", "w_sahara",
                                      "test_mse", "test_mae", "test_iou", "test_ssim"])
        for tag, r in results.items():
            table.add_data(tag, r["antarctica_weight"], r["sahara_weight"],
                           r["mse"], r["mae"], r["iou"], r["ssim"])
        wandb.log({"ablation/bias_mask_table": table})

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def load_or_generate_maps():
    cum_path = os.path.join(config.MAPS_DIR, "cumulative_maps.npy")
    if not os.path.exists(cum_path):
        print("KDE maps not found — generating…")
        df   = dp.load_and_clean()
        maps = kde_maps.generate_all_maps(df, save_npy=True)
        cum_maps = np.stack(
            [maps["cumulative"][i] for i in range(maps["num_decades"])], axis=0)
        inc_maps = np.stack(
            [maps["incremental"][i] for i in range(maps["num_decades"])], axis=0)
        bias_mask = maps["bias_mask"]
    else:
        print("Loading pre-generated KDE maps…")
        cum_maps, inc_maps = load_maps_from_disk()
        bias_mask = np.load(os.path.join(config.MAPS_DIR, "bias_mask.npy"))
    print(f"  cumulative {cum_maps.shape}  incremental {inc_maps.shape}")
    return cum_maps, inc_maps, bias_mask


def _log_prediction_image(model, test_dl, inc_maps, device, name="test_prediction"):
    """Log the test prediction alongside target as a wandb Image."""
    if not wandb_active():
        return
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            pred_np   = model(x.to(device))[0, 0].cpu().numpy()
            target_np = y[0, 0].cpu().numpy()
            break

    # Normalise to [0, 255] for display
    def to_img(arr):
        a = arr - arr.min()
        if a.max() > 0:
            a = a / a.max()
        return (a * 255).astype(np.uint8)

    wandb.log({
        f"{name}/prediction": wandb.Image(to_img(pred_np),
                                           caption="Model prediction"),
        f"{name}/target":     wandb.Image(to_img(target_np),
                                           caption="Ground truth"),
    })


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MeteoriteConvLSTM — Training Pipeline")
    print(f"  Device  : {device}")
    print(f"  Mode    : {args.model}")
    print(f"  Seed    : {args.seed}")
    print(f"{'='*60}\n")

    # ── W&B initialisation ────────────────────────────────────────────────────
    if not args.no_wandb:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)

        wandb.init(
            project=WANDB_PROJECT,
            name=f"full-pipeline-{args.model}" if not args.all_models
                  else "full-pipeline-all",
            config={
                # Data
                "seq_len":             args.seq_len,
                "batch_size":          args.batch_size,
                "train_decades":       config.TRAIN_END_DECADE,
                "val_decade":          config.VAL_DECADE,
                "test_decade":         config.TEST_DECADE,
                # Model
                "convlstm_hidden":     config.CONVLSTM_HIDDEN,
                "convlstm_kernel":     config.CONVLSTM_KERNEL,
                "decoder_channels":    config.DECODER_CHANNELS,
                "predrnn_hidden":      config.PREDRNN_HIDDEN,
                # Training
                "epochs":              args.epochs,
                "lr":                  args.lr,
                "weight_decay":        config.WEIGHT_DECAY,
                "loss_warmup_epochs":  config.LOSS_WARMUP_EPOCHS,
                # Loss
                "lambda_mse":          config.LAMBDA_MSE,
                "lambda_iou":          config.LAMBDA_IOU,
                "lambda_bias_mse":     config.LAMBDA_BIAS_MSE,
                "iou_threshold":       config.IOU_THRESHOLD,
                "warmup_epochs":       config.LOSS_WARMUP_EPOCHS,
                # Checkpoint
                "ckpt_iou_weight":     0.05,
            },
            tags=["full-pipeline"],
        )

    # ── Maps ──────────────────────────────────────────────────────────────────
    cum_maps, inc_maps, bias_mask = load_or_generate_maps()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl = build_dataloaders(
        cum_maps, inc_maps,
        seq_len=args.seq_len, batch_size=args.batch_size
    )

    loss_fn = CombinedLoss(bias_mask).to(device)

    # ── Select model(s) to train ──────────────────────────────────────────────
    MODEL_REGISTRY = {
        "convlstm":  (MeteoriteConvLSTM, {}),
        "attention": (AttentionConvLSTM,  {}),
        "predrnn":   (PredRNN,            {}),
    }

    if args.baselines_only:
        to_train = []
    elif args.all_models:
        to_train = list(MODEL_REGISTRY.keys())
    else:
        to_train = [args.model]

    all_test_results = {}
    best_model       = None
    best_model_name  = None

    for model_name in to_train:
        cls, kwargs = MODEL_REGISTRY[model_name]
        set_seed(args.seed)
        model = cls(**kwargs).to(device)

        if wandb_active():
            wandb.watch(model, log="gradients", log_freq=50, log_graph=(model_name == to_train[0]))
            _log({f"{model_name}/num_params": count_parameters(model)})

        print(f"\n{'─'*50}")
        print(f"  Training: {model_name.upper()}  ({count_parameters(model):,} params)")

        history, _ = train_model(
            model, train_dl, val_dl, args.epochs, device, bias_mask,
            lr=args.lr, use_warmup=True,
            tag=model_name,
            verbose=True,
            wandb_prefix=f"{model_name}/",
        )

        # Save checkpoint
        ckpt_path = os.path.join(config.CKPT_DIR, f"best_{model_name}.pt")
        torch.save({"state_dict": model.state_dict()}, ckpt_path)

        # Save training history for first / primary model
        if not to_train.index(model_name):
            history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
            with open(history_path, "w") as f_:
                json.dump(history, f_, indent=2)
            best_model      = model
            best_model_name = model_name

        test_metrics = evaluate(model, test_dl, loss_fn, device)
        all_test_results[model_name] = test_metrics
        print(f"  TEST → MSE={test_metrics['mse']:.4f}  "
              f"IoU={test_metrics['iou']:.4f}  SSIM={test_metrics['ssim']:.4f}")

        _log({f"{model_name}/test/mse":  test_metrics["mse"],
              f"{model_name}/test/mae":  test_metrics["mae"],
              f"{model_name}/test/iou":  test_metrics["iou"],
              f"{model_name}/test/ssim": test_metrics["ssim"]})

        _log_prediction_image(model, test_dl, inc_maps, device, name=model_name)

        # Save this model's own test-decade prediction (used by the figure
        # scripts, which load the file matching the model they display).
        model.eval()
        with torch.no_grad():
            for x, _ in test_dl:
                pred_np = model(x.to(device))[0, 0].cpu().numpy()
                break
        np.save(os.path.join(config.OUTPUT_DIR, f"test_prediction_{model_name}.npy"), pred_np)

    # ── Save the designated "showcase" prediction ───────────────────────────
    # The prediction-comparison and hotspot-map figures show AttentionConvLSTM;
    # fall back to the first trained model if attention wasn't part of this run.
    showcase_name = "attention" if "attention" in to_train else (
        to_train[0] if to_train else None)
    if showcase_name is not None:
        showcase_path = os.path.join(
            config.OUTPUT_DIR, f"test_prediction_{showcase_name}.npy")
        showcase_pred = np.load(showcase_path)
        np.save(os.path.join(config.OUTPUT_DIR, "test_prediction.npy"), showcase_pred)
        print(f"\nTest prediction saved: outputs/test_prediction.npy "
              f"(showcase model: {showcase_name})")

    # ── Climatological mean baseline ──────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Climatological Mean Baseline")
    clim_map     = climatological_mean_baseline(inc_maps, config.TRAIN_END_DECADE)
    clim_metrics = eval_static_prediction(clim_map, inc_maps[config.TEST_DECADE])
    all_test_results["ClimatologicalMean"] = clim_metrics
    print(f"  → MSE={clim_metrics['mse']:.4f}  IoU={clim_metrics['iou']:.4f}  "
          f"SSIM={clim_metrics['ssim']:.4f}")

    # ── Learned baselines ─────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Learned Baselines")
    BASELINE_REGISTRY = {
        "LastFrame":      LastFrameBaseline,
        "SingleFrameCNN": SingleFrameCNN,
        "LSTMFlat":       LSTMFlatBaseline,
    }
    for name, bl_cls in BASELINE_REGISTRY.items():
        set_seed(args.seed)
        bl_model = bl_cls().to(device)
        if name != "LastFrame":
            train_baseline_model(bl_model, train_dl, device, bias_mask,
                                  epochs=config.BASELINE_TRAIN_EPOCHS, tag=name)
        m = evaluate(bl_model, test_dl, loss_fn, device)
        all_test_results[name] = m
        print(f"  {name:<20s}: MSE={m['mse']:.4f}  IoU={m['iou']:.4f}  SSIM={m['ssim']:.4f}")
        _log({f"baseline/{name}/test_mse":  m["mse"],
              f"baseline/{name}/test_iou":  m["iou"],
              f"baseline/{name}/test_ssim": m["ssim"]})

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"  {'Method':<22s} {'MSE':>8} {'MAE':>8} {'IoU':>8} {'SSIM':>8}")
    print(f"  {'─'*50}")
    for name, m in all_test_results.items():
        marker = " ←" if name in MODEL_REGISTRY else ""
        print(f"  {name:<22s} {m['mse']:>8.4f} {m['mae']:>8.4f} "
              f"{m['iou']:>8.4f} {m['ssim']:>8.4f}{marker}")

    # Save results JSON
    results_path = os.path.join(config.OUTPUT_DIR, "test_results.json")
    with open(results_path, "w") as f_:
        json.dump(all_test_results, f_, indent=2)
    print(f"\nResults saved: {results_path}")

    # Log wandb comparison table
    if wandb_active():
        tbl = wandb.Table(columns=["method", "mse", "mae", "iou", "ssim"])
        for name, m in all_test_results.items():
            tbl.add_data(name, m["mse"], m["mae"], m["iou"], m["ssim"])
        wandb.log({"final_comparison_table": tbl})

    # ── Optional: rolling CV ───────────────────────────────────────────────────
    if args.rolling_cv and not args.baselines_only:
        all_cv_results = {}
        for model_name in to_train:
            cls, kwargs = MODEL_REGISTRY[model_name]
            cv_res, cv_sum = rolling_cv(
                cls, kwargs, cum_maps, inc_maps, bias_mask,
                test_decades=config.ROLLING_CV_TEST_DECADES,
                epochs=config.ROLLING_CV_EPOCHS,
                device=device, tag=model_name, seed=args.seed
            )
            all_cv_results[model_name] = {"folds": cv_res, "summary": cv_sum}
        cv_path = os.path.join(config.OUTPUT_DIR, "rolling_cv_results.json")
        with open(cv_path, "w") as f_:
            json.dump(all_cv_results, f_, indent=2)
        print(f"Rolling CV results saved: {cv_path}")

    # ── Optional: ablations ────────────────────────────────────────────────────
    if args.ablate and not args.baselines_only:
        ablation_results = {}
        ablate_list = (
            ["depth", "seqlen", "warmup", "loss", "bandwidth", "bias_mask"]
            if args.ablate == "all"
            else [args.ablate]
        )
        for abl in ablate_list:
            if abl == "depth":
                ablation_results["depth"] = ablation_depth(
                    cum_maps, inc_maps, bias_mask,
                    train_dl, val_dl, test_dl,
                    epochs=args.epochs, device=device, seed=args.seed)
            elif abl == "seqlen":
                ablation_results["seqlen"] = ablation_seqlen(
                    cum_maps, inc_maps, bias_mask, device, epochs=args.epochs,
                    seed=args.seed)
            elif abl == "warmup":
                ablation_results["warmup"] = ablation_warmup(
                    train_dl, val_dl, test_dl, bias_mask, device, epochs=args.epochs,
                    seed=args.seed)
            elif abl == "loss":
                ablation_results["loss_components"] = ablation_loss_components(
                    train_dl, val_dl, test_dl, bias_mask, device, epochs=args.epochs,
                    seed=args.seed)
            elif abl == "bandwidth":
                df_raw = dp.load_and_clean()
                ablation_results["bandwidth"] = ablation_bandwidth(
                    df_raw, bias_mask, epochs=args.epochs, device=device,
                    seed=args.seed)
            elif abl == "bias_mask":
                ablation_results["bias_mask"] = ablation_bias_mask(
                    train_dl, val_dl, test_dl, epochs=args.epochs, device=device,
                    seed=args.seed)
        abl_path = os.path.join(config.OUTPUT_DIR, "ablation_results.json")
        with open(abl_path, "w") as f_:
            json.dump(ablation_results, f_, indent=2)
        print(f"\nAblation results saved: {abl_path}")

    if wandb_active():
        wandb.finish()

    return best_model, all_test_results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MeteoriteConvLSTM family")
    p.add_argument("--model",          type=str,   default="convlstm",
                   choices=["convlstm", "attention", "predrnn"],
                   help="Which model to train (default: convlstm)")
    p.add_argument("--all-models",     action="store_true",
                   help="Train convlstm + attention + predrnn")
    p.add_argument("--baselines-only", action="store_true",
                   help="Only retrain learned baselines (SingleFrameCNN, LSTMFlat)")
    p.add_argument("--rolling-cv",     action="store_true",
                   help="Run rolling cross-validation after main training")
    p.add_argument("--ablate",         type=str, default=None,
                   choices=["depth", "seqlen", "warmup", "loss",
                            "bandwidth", "bias_mask", "all"],
                   help="Run ablation study")
    p.add_argument("--full-pipeline",  action="store_true",
                   help="Equivalent to --all-models --rolling-cv --ablate all")
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--lr",             type=float, default=config.LR)
    p.add_argument("--batch_size",     type=int,   default=config.BATCH_SIZE)
    p.add_argument("--seq_len",        type=int,   default=config.SEQ_LEN)
    p.add_argument("--seed",           type=int,   default=config.SEED,
                   help="Random seed — reset before every independent "
                        "training run for order-independent reproducibility")
    p.add_argument("--no-wandb",       action="store_true",
                   help="Disable Weights & Biases logging")
    p.add_argument("--wandb-key",      type=str,   default=None,
                   help="Optional W&B API key (overrides env/netrc)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.baselines_only = getattr(args, "baselines_only", False)

    # --full-pipeline expands to all sub-flags
    if args.full_pipeline:
        args.all_models = True
        args.rolling_cv = True
        args.ablate     = "all"

    train(args)
