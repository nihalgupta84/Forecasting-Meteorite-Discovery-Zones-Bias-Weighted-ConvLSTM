# Forecasting Meteorite Discovery Zones with Bias-Weighted ConvLSTM on Decadal KDE Density Map Sequences

Code, data, trained checkpoints and per-seed results for the paper

> N. Kumar, A. Kumar, Vidhi, D. Sharma, A. Tiwari, P. Gupta.
> *Forecasting Meteorite Discovery Zones with Bias-Weighted ConvLSTM on Decadal KDE Density Map Sequences.*
> 11th International Conference on Computer Vision and Image Processing (CVIP 2026), Springer.

## What this does

The NASA Meteorite Landings catalogue (38,048 usable records, 1800–2013) is turned into a sequence of
images and treated as a video-prediction problem:

1. **KDE encoding.** For each decade, find coordinates are converted into a Gaussian kernel density map on a
   2° grid (90 × 180). Two maps are built per decade: *incremental* (finds of that decade only) and
   *cumulative* (all finds up to that decade).
2. **Forecasting.** A two-layer ConvLSTM encoder–decoder takes the five previous cumulative maps and predicts
   the incremental map of the next decade. A CBAM-gated variant (AttentionConvLSTM) and PredRNN are included.
3. **Bias-weighted loss.** The loss is MSE + 0.1 · Soft-IoU + 0.5 · bias-weighted MSE. The weighting
   down-weights the heavily searched Antarctic (0.3) and Saharan (0.4) regions. The Soft-IoU term is switched
   on after a 20-epoch warmup.
4. **Hotspot map.** At inference, `U = prediction × mask` highlights predicted density outside the
   down-weighted collection regions.

The splits use decades 1850s–1990s for training (15 sequences), the 2000s for validation and 2010–2013 for
testing. A rolling cross-validation over the last four decades is also included. Every result is the
mean ± std over seeds 4, 5 and 6.

## Main result (test bin 2010–2013, mean ± std over 3 seeds)

| Method | MSE ↓ | MAE ↓ | SSIM ↑ | IoU ↑ |
|---|---|---|---|---|
| Climatological mean | .0045 | .0159 | .096 | .025 |
| Last-Frame (copy last cumulative map) | .0025 | **.0072** | **.378** | **.165** |
| Single-Frame CNN | **.0019 ± .0001** | .0278 ± .0009 | .142 ± .007 | .125 ± .003 |
| LSTM-Flat | .0025 ± .0002 | .0334 ± .0020 | .076 ± .007 | .006 ± .003 |
| PredRNN | .0029 ± .0004 | .0319 ± .0025 | .104 ± .009 | .139 ± .022 |
| ConvLSTM | .0027 ± .0011 | .0344 ± .0058 | .109 ± .048 | .152 ± .042 |
| AttentionConvLSTM | .0023 ± .0007 | .0312 ± .0008 | .111 ± .031 | .086 ± .022 |

Simple baselines are strong on this small dataset. The paper discusses why: cumulative inputs already contain
most of the next increment's structure, and the data issue described below. A loss ablation shows that removing
the Soft-IoU term lowers MSE (.0018 → .0007) and raises SSIM (.14 → .41).

## Known data issue: placeholder coordinates

**6,184 records (16.3%) in the catalogue have coordinates of exactly (0°, 0°).** This is a placeholder for
"location unknown", used mostly for Antarctic finds (Miller Range, Yamato, Dominion Range) and Northwest
Africa finds. They make up 58% of the 2010–2013 test bin.

These records are **kept** in all released experiments, to match the paper exactly. They create an artificial
density peak in the Gulf of Guinea that inflates IoU for every method. They also move recent Antarctic finds
away from Antarctica. `supplementary_analysis.py` quantifies the effect.

If you build on this code, we recommend filtering them out by adding this line to
`data_preprocessing.load_and_clean`:

```python
df = df[~((df["lat"] == 0) & (df["lon"] == 0))]
```

Then delete `outputs/kde_maps/` so the maps are regenerated, and retrain.

## Setup

```bash
conda create -n meteorite python=3.10 -y
conda activate meteorite
pip install -r requirements.txt
```

A CUDA GPU is recommended for training. Evaluation and the analysis scripts also run on CPU.
Weights & Biases is optional: pass `--no-wandb` to `train.py`. `run_multiseed.py` disables it by default.

## Quick start

**1. Check the released checkpoints** (CPU is fine, takes seconds):

```bash
python supplementary_analysis.py
```

This re-evaluates the seed-4 checkpoints and compares them with the seed-4 entries of
`outputs/multiseed_results.json`. Differences are below 1e-4 for ConvLSTM/AttentionConvLSTM, and around 2e-3
for PredRNN because of CPU vs GPU arithmetic. The script then runs the paper's single-seed diagnostics:
- IoU across thresholds;
- top-1% and top-5% hotspot overlap;
- metrics with the (0°, 0°) region excluded;
- catalogue statistics for each mask region;
- the top-5 hotspot coordinates;
- an error breakdown (share of MSE from Antarctic cells, MAE from empty target cells).

Output: `outputs/supplementary/supplementary_analysis.json`.

`python eval_from_checkpoints.py` prints the test metrics of the three checkpoints and writes
`outputs/checkpoint_eval.json`.

**2. Reproduce every table from scratch.** This trains all models, rolling CV and all ablations for 3 seeds:

```bash
python run_multiseed.py --epochs 200
```

Results are written to `outputs/multiseed_results.json` and `outputs/multiseed/seed{4,5,6}/`. **This overwrites
the released result files**; `git checkout outputs/` restores them. Seed 4 runs last, so its checkpoints and
predictions are the ones left in `outputs/`.

Single experiments:

```bash
python train.py --all-models --no-wandb            # ConvLSTM, AttentionConvLSTM, PredRNN + baselines
python train.py --model convlstm --rolling-cv --no-wandb
python train.py --ablate loss --no-wandb           # depth | seqlen | warmup | loss | bandwidth | bias_mask | all
```

**3. Figures**

```bash
python visualise.py      # original figures   → outputs/figures/
python make_figures.py   # map figures as used in the paper (opaque RGB, (0°,0°) marked)
```

## Repository layout

```
config.py               all hyperparameters, paths, grid, splits, bias-mask regions, ablation settings
data_preprocessing.py   loading and cleaning the catalogue, decade binning
kde_maps.py             KDE density maps and bias mask
dataset.py              sequence dataset and dataloaders (main split and rolling-CV folds)
model.py                ConvLSTM, AttentionConvLSTM (CBAM), PredRNN, Last-Frame / CNN / LSTM-Flat baselines
losses.py               MSE + Soft-IoU + bias-weighted MSE loss; evaluation metrics (MSE, MAE, IoU, SSIM)
train.py                training, evaluation, rolling CV, ablations
run_multiseed.py        full pipeline over seeds 4, 5, 6 and aggregation (mean ± std)
eval_from_checkpoints.py      re-evaluate the released checkpoints
supplementary_analysis.py     evaluation-only diagnostics reported in the paper
visualise.py, make_figures.py figure generation

dataset/Meteorite_Landings_cleaned.csv    NASA Meteorite Landings (38,115 rows; cleaned to 38,048 in code)
outputs/kde_maps/                         cumulative / incremental maps (22 × 90 × 180) and bias mask
outputs/checkpoints/                      seed-4 weights: best_convlstm.pt, best_attention.pt, best_predrnn.pt
outputs/test_prediction_*.npy             seed-4 test-bin predictions
outputs/multiseed_results.json            aggregated mean ± std for every table
outputs/multiseed/seed{4,5,6}/            per-seed main / rolling-CV / ablation results
outputs/supplementary/                    output of supplementary_analysis.py
```

Notes on the configuration:
- The reported models use loss weights (1.0, 0.1, 0.5) with a 20-epoch warmup.
- The loss-component ablation picks checkpoints by validation MSE. All other runs pick by the validation score
  MSE − 0.05 · IoU.
- The bias mask is used only in the loss and in the hotspot map; it is not applied to the network output.
- The third mask box (20–40° N, 80–120° E) contains only about 0.3% of records.

## Data

The NASA Meteorite Landings dataset is public
([data.nasa.gov](https://data.nasa.gov/Space-Science/Meteorite-Landings/gh4g-9sfh)), compiled by The
Meteoritical Society. A copy is included for reproducibility.

## Citation

```bibtex
@inproceedings{kumar2026meteorite,
  author    = {Kumar, Nihal and Kumar, Avishek and Vidhi and Sharma, Dhruv and Tiwari, Alok and Gupta, Piyush},
  title     = {Forecasting Meteorite Discovery Zones with Bias-Weighted {ConvLSTM} on Decadal {KDE} Density Map Sequences},
  booktitle = {Computer Vision and Image Processing (CVIP 2026)},
  publisher = {Springer},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
