"""
dataset.py
──────────
PyTorch Dataset and DataLoader builders for the spatiotemporal KDE map sequences.

Each sample = (input_sequence, target_map)
  input_sequence : Tensor of shape (SEQ_LEN, 1, H, W)  — T cumulative maps
  target_map     : Tensor of shape (1, H, W)            — next decade incremental map

Splits:
  Train  : decades [SEQ_LEN-1 .. TRAIN_END_DECADE-1]  (enough history to fill SEQ_LEN)
  Val    : decade  [VAL_DECADE]
  Test   : decade  [TEST_DECADE]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

import config


class MeteoriteDensityDataset(Dataset):
    """
    Loads pre-computed KDE maps and builds (sequence, target) pairs.

    Args:
        cumulative_maps  : np.ndarray (num_decades, H, W) — cumulative KDE maps
        incremental_maps : np.ndarray (num_decades, H, W) — incremental KDE maps
        decade_indices   : list[int]  — which decade indices to include as targets
        seq_len          : int        — number of past decades used as input
        augment          : bool       — whether to apply random horizontal flip
    """

    def __init__(self,
                 cumulative_maps:  np.ndarray,
                 incremental_maps: np.ndarray,
                 decade_indices:   list,
                 seq_len:          int  = config.SEQ_LEN,
                 augment:          bool = False):

        self.cum  = cumulative_maps.astype(np.float32)   # (D, H, W)
        self.inc  = incremental_maps.astype(np.float32)  # (D, H, W)
        self.indices = decade_indices
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        target_idx = self.indices[i]

        # Input: SEQ_LEN cumulative maps ending at (target_idx - 1)
        input_start = target_idx - self.seq_len
        input_end   = target_idx             # exclusive

        # Safety: if not enough history, pad with zeros from the left
        input_maps = []
        for t in range(input_start, input_end):
            if t < 0:
                input_maps.append(np.zeros((config.GRID_H, config.GRID_W),
                                           dtype=np.float32))
            else:
                input_maps.append(self.cum[t])

        # Stack → (SEQ_LEN, H, W) → add channel → (SEQ_LEN, 1, H, W)
        x = np.stack(input_maps, axis=0)[:, np.newaxis, :, :]  # (T, 1, H, W)

        # Target: incremental map at target_idx → (1, H, W)
        y = self.inc[target_idx][np.newaxis, :, :]              # (1, H, W)

        # Optional augmentation: horizontal flip (longitudinal)
        if self.augment and np.random.rand() > 0.5:
            x = x[:, :, :, ::-1].copy()
            y = y[:, :, ::-1].copy()

        return torch.from_numpy(x), torch.from_numpy(y)


def load_maps_from_disk():
    """Load all pre-generated .npy map stacks from disk."""
    cum_path = os.path.join(config.MAPS_DIR, "cumulative_maps.npy")
    inc_path = os.path.join(config.MAPS_DIR, "incremental_maps.npy")

    assert os.path.exists(cum_path), \
        f"Run kde_maps.py first — {cum_path} not found."
    assert os.path.exists(inc_path), \
        f"Run kde_maps.py first — {inc_path} not found."

    cum  = np.load(cum_path)   # (num_decades, H, W)
    inc  = np.load(inc_path)   # (num_decades, H, W)
    return cum, inc


def build_dataloaders(cum_maps:  np.ndarray = None,
                      inc_maps:  np.ndarray = None,
                      seq_len:   int        = config.SEQ_LEN,
                      batch_size: int       = config.BATCH_SIZE):
    """
    Build train / val / test DataLoaders.

    If cum_maps / inc_maps not supplied, loads from disk.
    """
    if cum_maps is None or inc_maps is None:
        cum_maps, inc_maps = load_maps_from_disk()

    num_decades = cum_maps.shape[0]

    # Train indices: from seq_len (need full history) up to TRAIN_END_DECADE
    train_indices = list(range(seq_len, config.TRAIN_END_DECADE + 1))

    # Val: single decade
    val_indices = [config.VAL_DECADE] if config.VAL_DECADE < num_decades else []

    # Test: single decade
    test_indices = [config.TEST_DECADE] if config.TEST_DECADE < num_decades else []

    print(f"Train decades: {len(train_indices)}  "
          f"({seq_len*10 + config.YEAR_START}s – "
          f"{(config.TRAIN_END_DECADE)*10 + config.YEAR_START}s)")
    print(f"Val   decade : {val_indices}")
    print(f"Test  decade : {test_indices}")

    train_ds = MeteoriteDensityDataset(cum_maps, inc_maps, train_indices,
                                       seq_len=seq_len, augment=True)
    val_ds   = MeteoriteDensityDataset(cum_maps, inc_maps, val_indices,
                                       seq_len=seq_len, augment=False)
    test_ds  = MeteoriteDensityDataset(cum_maps, inc_maps, test_indices,
                                       seq_len=seq_len, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def build_dataloaders_for_cv(cum_maps:         np.ndarray,
                              inc_maps:         np.ndarray,
                              train_end_decade: int,
                              val_decade:       int,
                              test_decade:      int,
                              seq_len:          int = config.SEQ_LEN,
                              batch_size:       int = config.BATCH_SIZE):
    """
    Build dataloaders for a single rolling-CV fold.

    train_end_decade : last decade index included in training
    val_decade       : single validation decade
    test_decade      : single test decade
    """
    train_indices = list(range(seq_len, train_end_decade + 1))
    val_indices   = [val_decade]
    test_indices  = [test_decade]

    train_ds = MeteoriteDensityDataset(cum_maps, inc_maps, train_indices,
                                       seq_len=seq_len, augment=True)
    val_ds   = MeteoriteDensityDataset(cum_maps, inc_maps, val_indices,
                                       seq_len=seq_len, augment=False)
    test_ds  = MeteoriteDensityDataset(cum_maps, inc_maps, test_indices,
                                       seq_len=seq_len, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    cum, inc = load_maps_from_disk()
    print(f"Cumulative  maps shape: {cum.shape}")
    print(f"Incremental maps shape: {inc.shape}")

    train_dl, val_dl, test_dl = build_dataloaders(cum, inc)

    # Inspect one batch
    for x, y in train_dl:
        print(f"\nBatch shapes — x: {x.shape}  y: {y.shape}")
        print(f"x value range: [{x.min():.3f}, {x.max():.3f}]")
        print(f"y value range: [{y.min():.3f}, {y.max():.3f}]")
        break
