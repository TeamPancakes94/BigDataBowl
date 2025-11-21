# 01_merge_pipeline.py
# create roster table from kaggle's train input/output

# placement

import pandas as pd
from pathlib import Path
from glob import glob

# TRAIN = Path("../train")
# OUT = Path("../outputs")
# OUT.mkdir(parents=True, exist_ok=True)

HERE  = Path(__file__).resolve().parent
TRAIN = (HERE / "../train").resolve()
OUT   = (HERE / "../outputs").resolve()
OUT.mkdir(parents=True, exist_ok=True)

def load_concat(pattern, limit=None):
    paths = sorted(TRAIN.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match {TRAIN.name}/{pattern}")
    if limit:
        paths = paths[:limit]
    dfs = [pd.read_csv(p, low_memory=False) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def main():
    inp = load_concat("input_2023_w*.csv", limit=2)
    _   = load_concat("output_2023_w*.csv", limit=2)  # presence check

    keep_cols = ["game_id", "play_id", "nfl_id",
                 "player_name", "player_position", "player_role", "player_side"]

    missing = [c for c in keep_cols if c not in inp.columns]
    if missing:
        raise KeyError(f"Missing columns in input files: {missing}")

    roster = (
        inp[keep_cols]
        .drop_duplicates()
        .sort_values(["game_id", "play_id", "nfl_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    out_path = OUT / "merged_core.csv"
    roster.to_csv(out_path, index=False)
    print("Merged core shape:", roster.shape)
    print("Wrote:", out_path.resolve())

if __name__ == "__main__":
    main()