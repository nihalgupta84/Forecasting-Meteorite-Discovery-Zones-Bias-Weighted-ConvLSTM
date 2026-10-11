"""
eval_from_checkpoints.py
────────────────────────
Re-evaluates the saved ConvLSTM / AttentionConvLSTM / PredRNN checkpoints
on the 2010-2013 test bin and writes outputs/checkpoint_eval.json
(the released test_results.json is left untouched).

    python eval_from_checkpoints.py
"""

import os, json
import numpy as np
import torch

import config
from dataset import build_dataloaders
from model import MeteoriteConvLSTM, AttentionConvLSTM, PredRNN
from losses import CombinedLoss, compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data ──────────────────────────────────────────────────────────────────
cum_maps  = np.load(os.path.join(config.MAPS_DIR, "cumulative_maps.npy"))
inc_maps  = np.load(os.path.join(config.MAPS_DIR, "incremental_maps.npy"))
bias_mask = np.load(os.path.join(config.MAPS_DIR, "bias_mask.npy"))

_, _, test_dl = build_dataloaders(cum_maps, inc_maps)
loss_fn = CombinedLoss(bias_mask).to(device)

MODELS = {
    "convlstm":  MeteoriteConvLSTM,
    "attention": AttentionConvLSTM,
    "predrnn":   PredRNN,
}

new_results = {}
for name, cls in MODELS.items():
    ckpt_path = os.path.join(config.CKPT_DIR, f"best_{name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  SKIP {name} — checkpoint not found")
        continue
    model = cls().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_m = {k: 0.0 for k in ["mse","mae","iou","ssim"]}
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            m = compute_metrics(pred, y)
            for k, v in m.items():
                all_m[k] += v
    n = max(len(test_dl), 1)
    avg = {k: v/n for k, v in all_m.items()}
    new_results[name] = avg
    print(f"  {name:<12s}: MSE={avg['mse']:.4f}  MAE={avg['mae']:.4f}  "
          f"IoU={avg['iou']:.4f}  SSIM={avg['ssim']:.4f}")

out_path = os.path.join(config.OUTPUT_DIR, "checkpoint_eval.json")
with open(out_path, "w") as f:
    json.dump(new_results, f, indent=2)
print(f"\nSaved → {out_path}")
