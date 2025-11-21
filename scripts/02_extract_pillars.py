# 02_extract_pillars.py
# compute simplified Sky Vision pillar metrics from tracking data

# placement

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# need for ball-in-air window
from _utils_data import load_weeks
from _utils_ball import find_throw_and_arrival, slice_ball_window, eyes_score

FPS = 10.0
MS_PER_FRAME = 1000.0 / FPS  # 100 ms/frame

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
OUT   = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# --- column normalization + robust loader ---
COLMAP = {
    "gameId": "game_id",
    "playId": "play_id",
    "nflId": "nfl_id",
    "frameId": "frame_id",
    "playerRole": "player_role",
}

REQUIRED = {
    "input":  ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "s", "o", "dir", "player_role"],
    "output": ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "player_role"],
}

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # lower-case everything first
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # then map kaggle camelCase -> snake_case used by the scripts
    df = df.rename(columns={k.lower(): v for k, v in COLMAP.items()})
    return df

# update load_concat
def load_concat(pattern, limit=None, kind="input"):
    paths = sorted(glob(str(TRAIN / pattern)))
    if not paths:
        raise FileNotFoundError(f"No files match train/{pattern}")
    if limit:
        paths = paths[:limit]

    dfs = [pd.read_csv(p, low_memory=False) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    # normalize column names (camelCase -> snake_case, lowercasing)
    df = _standardize_cols(df)

    base_cols = {"frame_id", "game_id", "play_id", "nfl_id", "x", "y"}
    missing = base_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {kind} files: {missing}. Have: {sorted(df.columns)}")

    # Only input files are expected to have player_role
    if kind == "input" and "player_role" not in df.columns:
        raise KeyError("Input files must contain 'player_role'")

    return df
# --- end replacement ---

def first_move_frame(g):
    m = g[g["s"] > 0.5]
    return m["frame_id"].min() if len(m) else np.nan

def circ_std_deg(series):
    """Circular std (degrees) for dir/o angles."""
    a = pd.Series(series).dropna().astype(float)
    if a.empty:
        return np.nan
    rad = np.deg2rad(a.to_numpy())
    C = np.mean(np.cos(rad)); S = np.mean(np.sin(rad))
    R = np.hypot(C, S)
    if R <= 0:
        return np.nan
    std_rad = np.sqrt(max(0.0, -2.0*np.log(R)))
    return np.rad2deg(std_rad)

def main():
    inp  = load_concat("input_2023_w*.csv",  limit=2, kind="input")
    outp = load_concat("output_2023_w*.csv", limit=2, kind="output")

    # Replace your old throw proxy / last-N logic for ball-in-air
    anchors = find_throw_and_arrival(inp, outp)

    # Slice to the true ball-in-air window for tracking-derived pillars:
    inp_window = slice_ball_window(inp, anchors)
    out_window = slice_ball_window(outp, anchors)

    # Compute EYES (1–10) per player in window
    # Choose which angle column you trust ('dir' or 'o')
    angle_col = "dir" if "dir" in inp_window.columns else "o"
    eyes = eyes_score(inp_window, angle_col=angle_col)  # columns: game_id, play_id, nfl_id, eyes_1_10
    # end of ball-in-air window implementation 

    # Attach player_role to output rows (output files usually lack it)
    roles = (
        inp[["game_id", "play_id", "nfl_id", "player_role"]]
            .dropna(subset=["player_role"])
            .drop_duplicates(subset=["game_id", "play_id", "nfl_id"], keep="last")
    )
    outp = outp.merge(roles, on=["game_id", "play_id", "nfl_id"], how="left")

    # anchors
    t0 = (inp.groupby(["game_id","play_id"])["frame_id"]
            .min().reset_index().rename(columns={"frame_id":"t0"}))
    t_throw = (inp.groupby(["game_id","play_id"])["frame_id"]
                 .max().reset_index().rename(columns={"frame_id":"t_throw"}))  # proxy

    # roles (case-insensitive)
    wr_in = inp[inp["player_role"].fillna("").str.contains("Targeted Receiver", case=False)].copy()
    db_in = inp[inp["player_role"].fillna("").str.contains("Defensive Coverage", case=False)].copy()

    # WR anticipation (ms)
    wr_move = (wr_in.groupby(["game_id","play_id","nfl_id"])
                    .apply(first_move_frame).reset_index(name="first_move"))
    wr_move = wr_move.merge(t0, on=["game_id","play_id"], how="left")
    wr_move["anticipation_ms"] = (wr_move["first_move"] - wr_move["t0"]) * MS_PER_FRAME

    # WR execution jitter (std of dy)
    wr_path = wr_in[["game_id","play_id","nfl_id","frame_id","y"]].copy()
    wr_path["dy"] = wr_path.groupby(["game_id","play_id","nfl_id"])["y"].diff()
    wr_jitter = (wr_path.groupby(["game_id","play_id","nfl_id"], as_index=False)["dy"]
                        .std().rename(columns={"dy":"execution_jitter"}))

    # WR innovation: circular std of heading in last N frames pre-throw
    N = 10
    last_in = wr_in.merge(t_throw, on=["game_id","play_id"], how="left")
    last_in = last_in[last_in["frame_id"] >= (last_in["t_throw"] - N)].copy()
    angle_col = "dir" if "dir" in last_in.columns else ("o" if "o" in last_in.columns else None)
    if angle_col is None:
        raise KeyError("Neither 'dir' nor 'o' found in input files.")
    wr_innov = (last_in.groupby(["game_id","play_id","nfl_id"])[angle_col]
                     .apply(circ_std_deg).reset_index(name="innovation_turn"))

    # earliest OUTPUT frame per play
    t1 = (outp.groupby(["game_id","play_id"])["frame_id"]
              .min().reset_index().rename(columns={"frame_id":"t1"}))
    o1 = outp.merge(t1, on=["game_id","play_id"], how="left")
    o1 = o1[o1["frame_id"] == o1["t1"]].copy()

    wr_o = o1[o1["player_role"].fillna("").str.contains("Targeted Receiver", case=False)].copy()
    db_o = o1[o1["player_role"].fillna("").str.contains("Defensive Coverage", case=False)].copy()

    wr_o = wr_o.rename(columns={"nfl_id":"wr_id","x":"x_wr","y":"y_wr"})
    db_o = db_o.rename(columns={"nfl_id":"db_id","x":"x_db","y":"y_db"})
    pair = wr_o.merge(db_o, on=["game_id","play_id"], how="inner")

    # distances and per-side selections (robust to empty)
    if not pair.empty:
        pair["dist"] = np.hypot(pair["x_wr"]-pair["x_db"], pair["y_wr"]-pair["y_db"])
        wr_sep = (pair.loc[pair.groupby(["game_id","play_id","wr_id"])["dist"].idxmin(),
                           ["game_id","play_id","wr_id","dist"]]
                       .rename(columns={"wr_id":"nfl_id","dist":"separation_yds"}))
        db_cov = (pair.loc[pair.groupby(["game_id","play_id"])["dist"].idxmin(),
                           ["game_id","play_id","db_id","dist"]]
                       .rename(columns={"db_id":"nfl_id","dist":"coverage_yds"}))
    else:
        wr_sep = pd.DataFrame(columns=["game_id","play_id","nfl_id","separation_yds"])
        db_cov = pd.DataFrame(columns=["game_id","play_id","nfl_id","coverage_yds"])

    # DB reaction (ms)
    db_move = (db_in.groupby(["game_id","play_id","nfl_id"])
                    .apply(first_move_frame).reset_index(name="first_move"))
    db_move = db_move.merge(t0, on=["game_id","play_id"], how="left")
    db_move["reaction_ms"] = (db_move["first_move"] - db_move["t0"]) * MS_PER_FRAME

    # assemble tidy features (snake_case keys)
    rows = []

    wr_feat = (wr_move[["game_id","play_id","nfl_id","anticipation_ms"]]
               .merge(wr_jitter, on=["game_id","play_id","nfl_id"], how="left")
               .merge(wr_innov,  on=["game_id","play_id","nfl_id"], how="left")
               .merge(wr_sep,    on=["game_id","play_id","nfl_id"], how="left"))
    
    # Add Eyes scores to WR feature table
    wr_feat = wr_feat.merge(eyes.rename(columns={"eyes_1_10": "eyes_score"}), on=["game_id", "play_id", "nfl_id"], how="left")


    for _, r in wr_feat.iterrows():
        rows += [
            {"game_id":r.game_id,"play_id":r.play_id,"player_id":r.nfl_id,"side":"WR","pillar":"anticipation","raw_value":r.anticipation_ms,"units":"ms"},
            {"game_id":r.game_id,"play_id":r.play_id,"player_id":r.nfl_id,"side":"WR","pillar":"execution","raw_value":r.execution_jitter,"units":"unit"},
            {"game_id":r.game_id,"play_id":r.play_id,"player_id":r.nfl_id,"side":"WR","pillar":"separation","raw_value":r.separation_yds,"units":"yd"},
            {"game_id":r.game_id,"play_id":r.play_id,"player_id":r.nfl_id,"side":"WR","pillar":"innovation","raw_value":r.innovation_turn,"units":"deg"},
            {"game_id": r.game_id, "play_id": r.play_id, "player_id": r.nfl_id, "side": "WR", "pillar": "eyes", "raw_value": r.eyes_score, "units": "score"},
        ]

    for _, r in db_move.iterrows():
        rows.append({"game_id":r.game_id,"play_id":r.play_id,"player_id":r.nfl_id,"side":"DB","pillar":"reaction","raw_value":r.reaction_ms,"units":"ms"})
    for _, r in db_cov.iterrows():
        rows.append({"game_id":r.game_id,"play_id":r.play_id,"player_id":r.nfl_id,"side":"DB","pillar":"coverage","raw_value":r.coverage_yds,"units":"yd"})

    feats = pd.DataFrame(rows, columns=["game_id","play_id","player_id","side","pillar","raw_value","units"])
    out_path = OUT / "merged_pillars.csv"
    feats.to_csv(out_path, index=False)
    print("Saved pillar features →", out_path.resolve(), "rows:", len(feats))

if __name__ == "__main__":
    main()