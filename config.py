"""
config.py — Central configuration for the entire pipeline.
All hyperparameters and paths in one place.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "dataset", "Meteorite_Landings_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MAPS_DIR   = os.path.join(OUTPUT_DIR, "kde_maps")
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")

for d in [OUTPUT_DIR, MAPS_DIR, CKPT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Grid resolution ────────────────────────────────────────────────────────────
# 2-degree resolution: 90 lat bins × 180 lon bins
GRID_H = 90    # latitude  bins  (-90  → +90  in 2° steps)
GRID_W = 180   # longitude bins  (-180 → +180 in 2° steps)

# ── Temporal settings ──────────────────────────────────────────────────────────
YEAR_START  = 1800   # first decade start
YEAR_END    = 2013   # dataset coverage cutoff
DECADE_SIZE = 10     # years per bin

# Train / Val / Test split (by decade index)
# Decades: 1800–1809 (idx 0), 1810–1819 (idx 1), ..., 2000–2009 (idx 20), 2010–2013 (idx 21)
TRAIN_END_DECADE = 19   # train on decades 0–19  (years 1800–1999)
VAL_DECADE       = 20   # validate on decade 20  (years 2000–2009)
TEST_DECADE      = 21   # test on decade 21       (years 2010–2013)

# Sequence length: how many past decades feed into ConvLSTM
SEQ_LEN = 5

# ── KDE bandwidth ──────────────────────────────────────────────────────────────
KDE_BANDWIDTH = 5.0    # degrees (Gaussian kernel σ in geographic space)
                       # Silverman's rule applied per decade; this is fallback

# ── Discovery Bias Mask ────────────────────────────────────────────────────────
# Regions dominated by organized collection programs — down-weighted in loss
BIAS_ZONES = [
    # (lat_min, lat_max, lon_min, lon_max, weight)
    (-90,  -70, -180, 180, 0.3),   # Antarctica (ANSMET program)
    ( 15,   35,    5,  55, 0.4),   # Sahara / Arabian Peninsula
    ( 20,   40,   80, 120, 0.4),   # South/East Asia box (holds ~0.3% of records)
]
UNBIASED_WEIGHT = 1.0   # weight for non-biased zones

# ── Model architecture ─────────────────────────────────────────────────────────
CONVLSTM_HIDDEN  = [32, 64]   # hidden channels per ConvLSTM layer
CONVLSTM_KERNEL  = 3          # spatial kernel size
DECODER_CHANNELS = [64, 32, 1]

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 1024
NUM_EPOCHS   = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4

# Loss weights
# NOTE: IoU loss alone at 0.5 caused the model to collapse to "predict nothing"
# (all-zero output has MSE≈0 and BiasWeightedMSE≈0, only IoU stays at ~0.8).
# Fix: reduce lambda_iou so MSE gradient dominates first.
LAMBDA_MSE      = 1.0
LAMBDA_IOU      = 0.1   # was 0.5 — reduced to prevent all-zero collapse
LAMBDA_BIAS_MSE = 0.5

IOU_THRESHOLD = 0.1   # binarize density maps at this value for IoU

# Warmup epochs: train with MSE-only loss before adding IoU term
# Set > 0 to avoid IoU collapse on cold-start
LOSS_WARMUP_EPOCHS = 20

# ── New model variants ─────────────────────────────────────────────────────────
ATTENTION_HIDDEN  = [32, 64]     # AttentionConvLSTM encoder channels (same grid)
PREDRNN_HIDDEN    = [32, 64]     # PredRNN spatiotemporal memory channels

# ── Rolling cross-validation ───────────────────────────────────────────────────
# Train on all decades before each test decade, evaluate on test decade.
# Gives multiple data points instead of a single test result.
ROLLING_CV_TEST_DECADES  = [18, 19, 20, 21]   # decade indices to hold out
ROLLING_CV_EPOCHS        = 150                 # shorter than main run for speed

# ── Ablation settings ──────────────────────────────────────────────────────────
# 1. Encoder depth: number of ConvLSTM layers
ABLATION_DEPTHS = [
    [32],           # 1 layer
    [32, 64],       # 2 layers (default)
    [32, 64, 128],  # 3 layers
]
# 2. Input sequence length
ABLATION_SEQ_LENS = [1, 3, 5, 7]
# 3. KDE bandwidth
ABLATION_KDE_BWS  = [3.0, 5.0, 8.0]   # degrees — 5.0 is default

# 4. Bias-mask weight sensitivity — (antarctica_weight, sahara/arabia_weight)
# "none"  = no down-weighting (uniform mask, sanity check)
# "loose" = mild down-weighting
# "ours"  = default, matches BIAS_ZONES above
# "tight" = aggressive down-weighting
ABLATION_BIAS_WEIGHTS = {
    "none":  (1.0, 1.0),
    "loose": (0.6, 0.7),
    "ours":  (0.3, 0.4),
    "tight": (0.15, 0.2),
}

# ── Baseline training ──────────────────────────────────────────────────────────
# Learned baselines (SingleFrameCNN, LSTMFlat) train for the same number of
# epochs as the main models for a fair comparison.
BASELINE_TRAIN_EPOCHS = 200

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 4