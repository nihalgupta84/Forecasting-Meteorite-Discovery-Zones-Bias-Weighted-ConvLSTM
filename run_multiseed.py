"""
run_multiseed.py
─────────────────
Runs the full experiment pipeline (main model comparison, rolling CV, and
every ablation) once per seed in SEEDS, then aggregates every reported
metric into outputs/multiseed_results.json as mean ± std across seeds.

This exists because train.py previously drew all randomness from a single
un-reset RNG stream shared across every sequential training call in a
process, so results silently depended on execution order (confirmed against
historical wandb logs). train.py now resets
the seed before every independent training run via train.set_seed(), so each
of the seeds below is run under identical, fully-specified conditions and
the only source of run-to-run variation is genuine training stochasticity
(weight init, unless models are seeded pinned per-config — see train.py).

Seed 4 (the pre-existing default seed) is run LAST so the checkpoints and
per-model prediction .npy files left on disk afterwards are seed 4's — these
remain the canonical artifacts used by every figure.

Usage:
    python run_multiseed.py                  # seeds 4, 5, 6 — 200 epochs
    python run_multiseed.py --epochs 200
"""

import argparse
import copy
import json
import os
import shutil
import statistics

import config
import train as T

SEEDS = [5, 6, 4]   # seed 4 last => its checkpoints/predictions are canonical
RESULT_FILES = ["test_results.json", "rolling_cv_results.json", "ablation_results.json"]
METRIC_KEYS = ["mse", "mae", "iou", "ssim"]


def run_one_seed(seed, epochs):
    args = argparse.Namespace(
        model="convlstm",
        all_models=True,
        baselines_only=False,
        rolling_cv=True,
        ablate="all",
        full_pipeline=False,
        epochs=epochs,
        lr=config.LR,
        batch_size=config.BATCH_SIZE,
        seq_len=config.SEQ_LEN,
        seed=seed,
        no_wandb=True,
        wandb_key=None,
    )
    print(f"\n{'#'*70}\n  MULTISEED RUN — seed={seed}\n{'#'*70}")
    T.train(args)

    # Snapshot this seed's result JSONs before the next seed's run
    # overwrites the shared output files.
    snap_dir = os.path.join(config.OUTPUT_DIR, "multiseed", f"seed{seed}")
    os.makedirs(snap_dir, exist_ok=True)
    for fname in RESULT_FILES:
        src = os.path.join(config.OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(snap_dir, fname))


def _mean_std(values):
    values = list(values)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(values), "per_seed": values}


def aggregate_test_results(per_seed_data, seeds):
    """test_results.json: {model_name: {metric: value, ...}}"""
    out = {}
    model_names = per_seed_data[0].keys()
    for name in model_names:
        out[name] = {}
        for metric in METRIC_KEYS:
            vals = [d[name][metric] for d in per_seed_data if metric in d.get(name, {})]
            if vals:
                out[name][metric] = _mean_std(vals)
    return out


def aggregate_rolling_cv(per_seed_data, seeds):
    """
    rolling_cv_results.json: {model_name: {folds: [...], summary: {...}}}
    Pools every individual fold across every seed (n_folds * n_seeds points
    per metric) rather than averaging the per-seed summaries, so the
    reported std reflects both temporal-fold variance and seed variance.
    """
    out = {}
    model_names = per_seed_data[0].keys()
    for name in model_names:
        pooled = {m: [] for m in METRIC_KEYS}
        for d in per_seed_data:
            for fold in d.get(name, {}).get("folds", []):
                for m in METRIC_KEYS:
                    if m in fold:
                        pooled[m].append(fold[m])
        out[name] = {m: _mean_std(v) for m, v in pooled.items() if v}
        out[name]["n_seeds"] = len(seeds)
        out[name]["n_folds_per_seed"] = len(per_seed_data[0].get(name, {}).get("folds", []))
    return out


def aggregate_ablation_results(per_seed_data, seeds):
    """
    ablation_results.json: {ablation_type: {variant_tag: {..., metric: value}}}
    Preserves non-metric metadata (channels, params, config, bandwidth, ...)
    from the first seed's run (identical by construction across seeds).
    """
    out = {}
    ablation_types = per_seed_data[0].keys()
    for abl_type in ablation_types:
        out[abl_type] = {}
        variant_tags = per_seed_data[0][abl_type].keys()
        for tag in variant_tags:
            variant_out = {}
            # Non-metric metadata from seed run 0
            for k, v in per_seed_data[0][abl_type][tag].items():
                if k not in METRIC_KEYS:
                    variant_out[k] = v
            for metric in METRIC_KEYS:
                vals = [d[abl_type][tag][metric] for d in per_seed_data
                        if metric in d.get(abl_type, {}).get(tag, {})]
                if vals:
                    variant_out[metric] = _mean_std(vals)
            out[abl_type][tag] = variant_out
    return out


def main():
    p = argparse.ArgumentParser(description="Run full pipeline across multiple seeds")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args()

    for seed in args.seeds:
        run_one_seed(seed, epochs=args.epochs)

    # ── Aggregate ──────────────────────────────────────────────────────────
    per_seed = {}
    for fname in RESULT_FILES:
        per_seed[fname] = []
        for seed in args.seeds:
            path = os.path.join(config.OUTPUT_DIR, "multiseed", f"seed{seed}", fname)
            with open(path) as f:
                per_seed[fname].append(json.load(f))

    aggregated = {
        "seeds": args.seeds,
        "epochs": args.epochs,
        "main_comparison": aggregate_test_results(per_seed["test_results.json"], args.seeds),
        "rolling_cv": aggregate_rolling_cv(per_seed["rolling_cv_results.json"], args.seeds),
        "ablations": aggregate_ablation_results(per_seed["ablation_results.json"], args.seeds),
    }

    out_path = os.path.join(config.OUTPUT_DIR, "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nMulti-seed aggregated results saved: {out_path}")


if __name__ == "__main__":
    main()
