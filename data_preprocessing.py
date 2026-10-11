"""
data_preprocessing.py
─────────────────────
Loads and cleans the NASA Meteorite Landings dataset.
Produces decade-binned coordinate lists ready for KDE map generation.

Run standalone:  python data_preprocessing.py
"""

import pandas as pd
import numpy as np
import config


def load_and_clean(path: str = config.DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned CSV and apply all quality filters.

    Filters applied:
      - Drop rows with missing lat/long
      - Drop rows with year outside [YEAR_START, YEAR_END]
      - Drop rows with mass == 0
      - Reset index
    """
    df = pd.read_csv(path)

    print(f"Raw dataset: {len(df):,} rows")

    # 1. Standardise column names (lowercase, strip spaces)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 2. Rename for convenience
    rename_map = {
        "mass_(g)":  "mass_g",
        "reclat":    "lat",
        "reclong":   "lon",
        "geolocation": "geo",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)

    # 3. Drop rows with no coordinates
    before = len(df)
    df.dropna(subset=["lat", "lon"], inplace=True)
    print(f"  After dropping missing coords:  {len(df):,} rows  "
          f"(removed {before - len(df):,})")

    # 4. Drop rows with invalid year
    before = len(df)
    df = df[df["year"].notna()]
    df = df[(df["year"] >= config.YEAR_START) & (df["year"] <= config.YEAR_END)]
    print(f"  After year filter [{config.YEAR_START}–{config.YEAR_END}]:  "
          f"{len(df):,} rows  (removed {before - len(df):,})")

    # 5. Drop zero-mass records
    before = len(df)
    df = df[df["mass_g"] > 0]
    print(f"  After mass > 0 filter:          {len(df):,} rows  "
          f"(removed {before - len(df):,})")

    # 6. Clip lat/lon to valid ranges
    df["lat"] = df["lat"].clip(-90, 90)
    df["lon"] = df["lon"].clip(-180, 180)

    # 7. Add decade column
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    df["decade_idx"] = (
        (df["decade"] - config.YEAR_START) // config.DECADE_SIZE
    ).astype(int)

    df.reset_index(drop=True, inplace=True)
    print(f"\nFinal clean dataset: {len(df):,} rows")
    print(f"  Fell:  {(df['fall'] == 'Fell').sum():,}")
    print(f"  Found: {(df['fall'] == 'Found').sum():,}")
    print(f"  Decade range: {df['decade'].min()} – {df['decade'].max()}")
    print(f"  Decade indices: 0 – {df['decade_idx'].max()}")

    return df


def get_decade_coords(df: pd.DataFrame) -> dict:
    """
    Returns a dict:  {decade_idx: np.ndarray of shape (N, 2) [lat, lon]}
    Covers all decades from 0 to max_idx, even empty ones.
    """
    max_idx = int((config.YEAR_END - config.YEAR_START) // config.DECADE_SIZE)
    coords_by_decade = {}

    for idx in range(max_idx + 1):
        subset = df[df["decade_idx"] == idx][["lat", "lon"]].values
        coords_by_decade[idx] = subset  # shape (N, 2) or (0, 2)

    return coords_by_decade


def get_cumulative_coords(df: pd.DataFrame) -> dict:
    """
    Returns cumulative coordinate arrays up to each decade:
      {decade_idx: np.ndarray of shape (N_cumulative, 2)}
    """
    max_idx = int((config.YEAR_END - config.YEAR_START) // config.DECADE_SIZE)
    cumulative = {}
    running = []

    for idx in range(max_idx + 1):
        subset = df[df["decade_idx"] == idx][["lat", "lon"]].values
        if len(subset) > 0:
            running.append(subset)
        if running:
            cumulative[idx] = np.vstack(running)
        else:
            cumulative[idx] = np.empty((0, 2))

    return cumulative


def decade_idx_to_year(idx: int) -> int:
    return config.YEAR_START + idx * config.DECADE_SIZE


def print_decade_stats(df: pd.DataFrame) -> None:
    print("\nDecade-wise landing counts:")
    print("-" * 40)
    g = df.groupby("decade").size().reset_index(name="count")
    for _, row in g.iterrows():
        bar = "█" * min(int(row["count"] / 50), 60)
        print(f"  {int(row['decade'])}s: {row['count']:5d}  {bar}")


if __name__ == "__main__":
    df = load_and_clean()
    print_decade_stats(df)
    coords = get_decade_coords(df)
    print(f"\nSample — decade idx 20 (2000s): "
          f"{len(coords[20])} records")
