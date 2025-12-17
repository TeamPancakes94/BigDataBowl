# %% [markdown]
# # %% [markdown]
# # # Sky Vision: Ball-in-Air PER-10 Evaluation
# #
# # This notebook is the unified version of our Big Data Bowl submission.
# #
# # What happens here:
# #
# # We locate the preprocessed tracking data (`input_2023_w*.csv` / `output_2023_w*.csv`)
# #   without hard-coding a Kaggle dataset name.
# # We build a **merged core roster** (player–play identities).
# # We define a robust **ball-in-air window** using ball-tracking rows when available,
# #   with safe fallbacks when things are missing.
# # We compute **PER-10 pillars** (Anticipation, Separation, Execution, Eyes,
# #   Innovation, Improv) from those ball-in-air frames.
# # - We derive WR/DB pillar features (anticipation ms, reaction ms, separation, coverage, eyes)
# #   and then run a **beta–binomial model** to get posterior distributions and
# #   0–100 ratings.
# # - Along the way, we show tables and a few simple plots so the workflow is
# #   understandable, not just a wall of code.

# %% [markdown]
# # %% [markdown]
# # ## 0. Imports and setup

# %%
# %%
import os
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def find_train_dir():
    """
    Find the `train/` directory that holds input_*.csv and output_*.csv.

    Priority:
      1. Kaggle: /kaggle/input/*/train
      2. Local: walk up from the current working directory and look for 'train/'
    """

    # ---------- 1) Kaggle environment ----------
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        candidates = []
        for ds in kaggle_root.iterdir():
            if not ds.is_dir():
                continue
            t = ds / "train"
            if t.is_dir():
                has_input = list(t.glob("input_2023_w*.csv")) or list(t.glob("input_*.csv"))
                has_output = list(t.glob("output_2023_w*.csv")) or list(t.glob("output_*.csv"))
                if has_input and has_output:
                    candidates.append(t)

        if candidates:
            print("Using TRAIN directory from Kaggle input:", candidates[0])
            return candidates[0]

    # ---------- 2) Local environment (VS Code, etc.) ----------
    # Start from current working directory and walk up the parents
    start = Path.cwd().resolve()
    for parent in [start] + list(start.parents):
        train_dir = parent / "train"
        if train_dir.is_dir():
            has_input = list(train_dir.glob("input_2023_w*.csv")) or list(train_dir.glob("input_*.csv"))
            has_output = list(train_dir.glob("output_2023_w*.csv")) or list(train_dir.glob("output_*.csv"))
            if has_input and has_output:
                print("Using TRAIN directory from local filesystem:", train_dir)
                return train_dir

    # If nothing found, give a clear error
    raise FileNotFoundError(
        "Could not find a 'train' directory with input_*.csv and output_*.csv.\n"
        "- On Kaggle: make sure you attached the dataset that contains train/.\n"
        "- Locally: make sure there's a 'train' folder somewhere above this notebook."
    )


TRAIN = find_train_dir()

# All outputs go here (works on Kaggle and locally)
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

print("TRAIN:", TRAIN.resolve())
print("OUT:  ", OUT.resolve())

# %% [markdown]
# # %% [markdown]
# # ## 1. Quick look at raw data
# #
# # For context, we briefly inspect:
# # What files exist in `train/`.
# # Standard Big Data Bowl tables (`players.csv`, `plays.csv`, `games.csv`) if we can find them.

# %%
# %%
# List CSV files in TRAIN
all_csvs = sorted(TRAIN.glob("*.csv"))
print("Found", len(all_csvs), "CSV files in TRAIN:")
for p in all_csvs[:20]:
    print(" -", p.name)
if len(all_csvs) > 20:
    print(" ...")


def find_and_load_single_csv(name: str):
    """
    Try to locate a CSV by filename anywhere under /kaggle/input.
    This is purely for exploratory display and not required for the main pipeline.
    """
    root = Path("/kaggle/input")
    if not root.exists():
        return None

    for ds in root.iterdir():
        if not ds.is_dir():
            continue
        candidate = ds / name
        if candidate.exists():
            print(f"Loading {name} from {candidate}")
            return pd.read_csv(candidate, low_memory=False)
    print(f"{name} not found in /kaggle/input (skipping)")
    return None


players = find_and_load_single_csv("players.csv")
plays   = find_and_load_single_csv("plays.csv")
games   = find_and_load_single_csv("games.csv")

if players is not None:
    display(players.head())
if plays is not None:
    display(plays.head())
if games is not None:
    display(games.head())


# %%
# %%
# Simple passResult distribution if available
if plays is not None and "passResult" in plays.columns:
    print(plays["passResult"].value_counts(dropna=False))

    plays["passResult"].value_counts().plot(kind="bar")
    plt.title("Distribution of passResult")
    plt.xlabel("Result")
    plt.ylabel("Count")
    plt.show()

# %% [markdown]
# # %% [markdown]
# # ## 2. Ball-in-air helpers
# #
# # A lot of our evaluation focuses on the window between the throw and arrival.
# # We approximate that window using the ball-tracking rows (where `team == "football"`).
# #
# # If those rows exist, we use the first and last ball frame as throw/arrival.
# # If they are missing, we fall back to the full output frame range per play.
# #
# # This makes the logic robust to missing or noisy event tags.

# %%
# %%
def find_throw_and_arrival(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (game_id, play_id), estimate t_throw and t_arrival.

    Preferred:
      - Use rows with team == 'football' in the output tracking and take
        the first and last frame_id as throw and arrival.

    Fallback:
      - If ball rows don't exist, use min and max frame_id from the output
        data for that play.

    All we need from output_df is: game_id, play_id, frame_id, and optionally team.
    """
    ball_df = None
    if "team" in output_df.columns:
        ball_df = output_df[output_df["team"].astype(str).str.lower() == "football"].copy()

    if ball_df is not None and not ball_df.empty:
        t_throw = (
            ball_df.groupby(["game_id", "play_id"])["frame_id"]
            .min()
            .reset_index()
            .rename(columns={"frame_id": "t_throw"})
        )
        t_arrival = (
            ball_df.groupby(["game_id", "play_id"])["frame_id"]
            .max()
            .reset_index()
            .rename(columns={"frame_id": "t_arrival"})
        )
        anchors = t_throw.merge(t_arrival, on=["game_id", "play_id"], how="inner")
    else:
        anchors = (
            output_df.groupby(["game_id", "play_id"])["frame_id"]
            .agg(t_throw="min", t_arrival="max")
            .reset_index()
        )
    return anchors


def slice_ball_window(df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only frames where frame_id is between [t_throw, t_arrival] for each play.
    """
    merged = df.merge(anchors, on=["game_id", "play_id"], how="inner")
    window = merged.query("frame_id >= t_throw and frame_id <= t_arrival").copy()
    return window


def compute_ball_landing(output_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate where the ball 'lands' for each play using the ball rows.

    We take the last ball-tracking frame for each (game_id, play_id)
    and treat its (x, y) as (ball_land_x, ball_land_y).

    If ball rows are missing for a play, that play simply won't appear in
    the returned dataframe, and later computations will see NaNs.
    """
    if "team" not in output_df.columns:
        return pd.DataFrame(columns=["game_id", "play_id", "ball_land_x", "ball_land_y"])

    ball_df = output_df[output_df["team"].astype(str).str.lower() == "football"].copy()
    if ball_df.empty:
        return pd.DataFrame(columns=["game_id", "play_id", "ball_land_x", "ball_land_y"])

    last_ball = (
        ball_df.sort_values("frame_id")
        .groupby(["game_id", "play_id"], as_index=False)
        .tail(1)[["game_id", "play_id", "x", "y"]]
        .rename(columns={"x": "ball_land_x", "y": "ball_land_y"})
    )
    return last_ball

# %% [markdown]
# # %% [markdown]
# # ## 3. PER-10 pillar functions'
# #
# # These functions define:
# #
# # `compute_all_anticipation`
# # `compute_all_separation`
# # `compute_all_execution`
# # `compute_all_eyes`
# # `compute_all_innovation`
# # `compute_all_improv`
# #  and the helpers for PER-10 / PER-10 360.
# #
# 

# %%
# %%
# --- BEGIN: per10.py -----------------------------------------------------------------

import numpy as np  # re-import here; harmless duplication
import pandas as pd  # re-import here; harmless duplication

# PER-10 (5-pillars) and PER-10 360 (6-pillars)

def compute_per10_basic(A, S, E, Eyes, Innovation):
    """
    5-pillar PER-10:
    PER-10 = average(A, S, E, Eyes, Innovation) on a 1–10 scale.
    """
    vals = np.array([A, S, E, Eyes, Innovation], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def compute_per10_360(A, S, E, Eyes, Innovation, Improv):
    """
    6-pillar PER-10 360:
    Includes improv as an additional dimension.
    """
    vals = np.array([A, S, E, Eyes, Innovation, Improv], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def compute_all_per10_360(
    A_df: pd.DataFrame,
    S_df: pd.DataFrame,
    E_df: pd.DataFrame,
    Eyes_df: pd.DataFrame,
    Innovation_df: pd.DataFrame,
    Improv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all pillar dataframes and compute PER-10 and PER-10 360.
    """
    df = (
        A_df
        .merge(S_df,    on=["game_id", "play_id", "nfl_id"], how="outer")
        .merge(E_df,    on=["game_id", "play_id", "nfl_id"], how="outer")
        .merge(
            Eyes_df.rename(columns={"eyes_score": "Eyes"}),
            on=["game_id", "play_id", "nfl_id"],
            how="outer",
        )
        .merge(
            Innovation_df.rename(columns={"innovation_score": "Innovation"}),
            on=["game_id", "play_id", "nfl_id"],
            how="outer",
        )
        .merge(
            Improv_df.rename(columns={"improv_score": "Improv"}),
            on=["game_id", "play_id", "nfl_id"],
            how="outer",
        )
    )

    df["PER10"] = df.apply(
        lambda row: compute_per10_basic(
            row["A"], row["S"], row["E"], row["Eyes"], row["Innovation"]
        ),
        axis=1,
    )

    df["PER10_360"] = df.apply(
        lambda row: compute_per10_360(
            row["A"], row["S"], row["E"], row["Eyes"], row["Innovation"], row["Improv"]
        ),
        axis=1,
    )

    return df

# Anticipation (A) -------------------------------------------

def compute_anticipation_for_group(g):
    """
    g: dataframe with one (game_id, play_id, nfl_id)
    Must have x, y, frame_id, ball_land_x, ball_land_y.
    """
    g = g.sort_values("frame_id").copy()

    g["dx"] = g["x"].diff()
    g["dy"] = g["y"].diff()

    g["bx"] = g["ball_land_x"] - g["x"]
    g["by"] = g["ball_land_y"] - g["y"]

    g["alignment"] = g["dx"] * g["bx"] + g["dy"] * g["by"]

    g["alignment_norm"] = (
        g["alignment"] /
        (np.sqrt(g["dx"]**2 + g["dy"]**2) * np.sqrt(g["bx"]**2 + g["by"]**2) + 1e-6)
    )

    g = g.replace([np.inf, -np.inf], np.nan)

    if g["alignment_norm"].isna().all():
        return np.nan

    first_valid = g["alignment_norm"].dropna().iloc[0]
    A = int(np.round(np.clip(first_valid * 10, 0, 10)))
    return A


def compute_all_anticipation(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: game_id, play_id, nfl_id, A
    """
    results = []
    for (game_id, play_id, nfl_id), g in df_input.groupby(["game_id", "play_id", "nfl_id"]):
        A = compute_anticipation_for_group(g)
        results.append([game_id, play_id, nfl_id, A])

    return pd.DataFrame(results, columns=["game_id", "play_id", "nfl_id", "A"])



# Separation (S) ------------------------------------

def score_separation(delta_sep, role):
    """
    WR (Offense): positive delta = gained separation = good
    DB (Defense): negative delta = closed separation = good
    """
    if role == "Offense":
        if delta_sep >= 3:   return 10
        if delta_sep >= 2:   return 8
        if delta_sep >= 1:   return 6
        if delta_sep >= 0.5: return 5
        return 3
    else:
        # DB: we like closing separation
        if delta_sep <= -3:  return 10
        if delta_sep <= -2:  return 8
        if delta_sep <= -1:  return 6
        if delta_sep <= -0.5:return 5
        return 3


def compute_player_separation(play_df: pd.DataFrame, player_id) -> float:
    """
    Computes a player's separation score for one play.
    Works for WRs (Offense) and DBs (Defense).
    """
    player_rows = play_df[play_df["nfl_id"] == player_id].sort_values("frame_id")
    if len(player_rows) < 2:
        return None

    role = player_rows["player_side"].iloc[0]

    opp_df = play_df[play_df["player_side"] != role].copy()
    if opp_df.empty:
        return None

    separations = []
    for _, p in player_rows.iterrows():
        frame = p["frame_id"]
        opp_same_frame = opp_df[opp_df["frame_id"] == frame]
        if opp_same_frame.empty:
            continue
        dists = np.sqrt(
            (opp_same_frame["x"] - p["x"]) ** 2 +
            (opp_same_frame["y"] - p["y"]) ** 2
        )
        separations.append(dists.min())

    if len(separations) < 2:
        return None

    start_sep = separations[0]
    end_sep = separations[-1]
    delta_sep = end_sep - start_sep

    return score_separation(delta_sep, role)


def compute_all_separation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: game_id, play_id, nfl_id, S
    """
    results = []
    for (game_id, play_id), play_df in df.groupby(["game_id", "play_id"]):
        for player_id in play_df["nfl_id"].unique():
            S = compute_player_separation(play_df, player_id)
            if S is not None:
                results.append([game_id, play_id, player_id, S])

    return pd.DataFrame(results, columns=["game_id", "play_id", "nfl_id", "S"])



# Execution (E) ---------------------------------------------

def angle_diff(a, b):
    """
    Smallest difference between two angles in degrees.
    """
    d = (a - b + 180) % 360 - 180
    return abs(d)


def compute_execution_for_group(g: pd.DataFrame) -> float:
    """
    Execution score (0–10) based on jerk (smoothness), alignment,
    and path smoothness.
    """
    g = g.sort_values("frame_id").copy()
    if len(g) < 3:
        return np.nan

    if "dir" not in g.columns or "o" not in g.columns:
        return np.nan

    # Jerk / instability in direction
    directions = g["dir"].to_numpy()
    dir_change = np.abs((np.diff(directions) + 180) % 360 - 180)
    if len(dir_change) == 0:
        return np.nan
    jerk = np.mean(dir_change)
    E1 = 1 / (1 + jerk)

    # Orientation vs movement alignment
    misalign = [
        angle_diff(g["dir"].iloc[i], g["o"].iloc[i])
        for i in range(len(g))
    ]
    misalign_norm = np.mean(misalign) / 180
    E2 = 1 - misalign_norm

    # Path smoothness
    dx = g["x"].diff().fillna(0)
    dy = g["y"].diff().fillna(0)
    path_angles = np.degrees(np.arctan2(dy, dx)) % 360
    if len(path_angles) < 3:
        return np.nan
    path_diff = np.abs((np.diff(path_angles) + 180) % 360 - 180)
    if len(path_diff) == 0:
        return np.nan
    path_smoothness = 1 - (np.mean(path_diff) / 180)
    E3 = path_smoothness

    raw_E = np.mean([E1, E2, E3])
    E_score = int(np.round(raw_E * 10))
    return E_score


def compute_all_execution(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: game_id, play_id, nfl_id, E
    """
    results = []
    for (game_id, play_id, nfl_id), g in df_input.groupby(["game_id", "play_id", "nfl_id"]):
        E = compute_execution_for_group(g)
        results.append([game_id, play_id, nfl_id, E])

    return pd.DataFrame(results, columns=["game_id", "play_id", "nfl_id", "E"])


# Eyes (tracking / vision) -------------------------------------------

def compute_eyes_score(play_df: pd.DataFrame, landing_point, player_id) -> float:
    """
    Eyes score on a 0–10 scale for a single (game_id, play_id, nfl_id).
    """
    px, py = landing_point

    df = play_df[play_df["nfl_id"] == player_id].copy()
    if df.empty:
        return np.nan

    dx = px - df["x"]
    dy = py - df["y"]

    ball_angle = np.degrees(np.arctan2(dy, dx)) % 360
    orientation = df["o"] % 360

    angle_diff_local = np.abs((ball_angle - orientation + 180) % 360 - 180)
    misalign = np.mean(angle_diff_local)

    # Perfect tracking ~ 0°, bad ~ 180°
    score = 10 * (1 - misalign / 180)
    return max(0, min(10, score))


def compute_all_eyes(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: ball-in-air frames with columns:
        game_id, play_id, nfl_id, x, y, o, ball_land_x, ball_land_y

    Returns: game_id, play_id, nfl_id, eyes_score
    """
    results = []
    for (game_id, play_id), play_df in df.groupby(["game_id", "play_id"]):
        lp = play_df.dropna(subset=["ball_land_x", "ball_land_y"])
        if lp.empty:
            continue
        ball_land_x = float(lp["ball_land_x"].iloc[0])
        ball_land_y = float(lp["ball_land_y"].iloc[0])
        landing_point = (ball_land_x, ball_land_y)

        for player_id in play_df["nfl_id"].unique():
            score = compute_eyes_score(play_df, landing_point, player_id)
            results.append({
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": player_id,
                "eyes_score": score,
            })

    return pd.DataFrame(results)


# Innovation --------------------------------------------------------

def compute_innovation_for_group(df: pd.DataFrame) -> float:
    """
    Innovation based on how much the player's path deviates from a
    straight-line fit, and how much they change direction.
    """
    df = df.sort_values("frame_id").copy()
    if len(df) < 3:
        return np.nan

    dx = df["x"].diff()
    dy = df["y"].diff()

    directions = np.degrees(np.arctan2(dy, dx)) % 360
    direction_change = np.abs((directions.diff() + 180) % 360 - 180)
    mean_cut_angle = np.nanmean(direction_change[1:])  # skip first NaN

    frames = df["frame_id"].values
    coef_x = np.polyfit(frames, df["x"], 1)
    coef_y = np.polyfit(frames, df["y"], 1)
    pred_x = np.polyval(coef_x, frames)
    pred_y = np.polyval(coef_y, frames)
    deviation = np.sqrt((pred_x - df["x"]) ** 2 + (pred_y - df["y"]) ** 2)
    mean_deviation = np.mean(deviation)

    raw = 0.5 * (mean_cut_angle / 90) + 0.5 * (mean_deviation / 3)
    score = 10 * np.clip(raw, 0, 1)
    return float(score)


def compute_all_innovation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: game_id, play_id, nfl_id, innovation_score
    """
    results = []
    for (game_id, play_id, nfl_id), g in df.groupby(["game_id", "play_id", "nfl_id"]):
        score = compute_innovation_for_group(g)
        results.append({
            "game_id": game_id,
            "play_id": play_id,
            "nfl_id": nfl_id,
            "innovation_score": score,
        })

    return pd.DataFrame(results)


# Improv ------------------------------------------------------------------

def compute_improv_score(play_df: pd.DataFrame, player_id, role: str) -> float:
    df = play_df[play_df["nfl_id"] == player_id].copy()
    if df.empty or len(df) < 5:
        return np.nan

    speed = df["s"].values
    speed_diff = np.abs(np.diff(speed))

    dx = df["x"].diff().fillna(0)
    dy = df["y"].diff().fillna(0)

    directions = np.degrees(np.arctan2(dy, dx)) % 360
    dir_diff = np.abs((np.diff(directions) + 180) % 360 - 180)

    disruption_raw = np.mean(speed_diff) + (np.mean(dir_diff) / 2.0)
    D = np.clip(disruption_raw / 5.0, 0, 10)

    # Recoverability: how fast the player settles down after a big change
    post = np.abs(dir_diff)
    if len(post) < 2:
        R = 5.0
    else:
        if (post < 10).any():
            recover_time = int(np.argmax(post < 10))
        else:
            recover_time = len(post)
        R = np.clip(10 - recover_time * 1.5, 0, 10)

    G = 0.0
    if role == "Offense":
        if "ball_land_x" in df.columns and "ball_land_y" in df.columns:
            lx = df["ball_land_x"].iloc[0]
            ly = df["ball_land_y"].iloc[0]
            initial_dist = np.sqrt((df["x"].iloc[0] - lx) ** 2 + (df["y"].iloc[0] - ly) ** 2)
            final_dist = np.sqrt((df["x"].iloc[-1] - lx) ** 2 + (df["y"].iloc[-1] - ly) ** 2)
            G = np.clip(initial_dist - final_dist, 0, 10)
    else:
        if "target_separation" in df.columns:
            sep0 = df["target_separation"].iloc[0]
            sep1 = df["target_separation"].iloc[-1]
            G = np.clip(sep0 - sep1, 0, 10)

    improv = 0.4 * D + 0.3 * R + 0.3 * G
    return round(float(improv), 2)


def compute_all_improv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: game_id, play_id, nfl_id, improv_score
    """
    results = []
    for (game_id, play_id), play_df in df.groupby(["game_id", "play_id"]):
        for player_id in play_df["nfl_id"].unique():
            player_rows = play_df[play_df["nfl_id"] == player_id]
            if player_rows.empty:
                continue
            role = player_rows["player_side"].iloc[0]
            score = compute_improv_score(play_df, player_id, role)
            results.append({
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": player_id,
                "improv_score": score,
            })

    return pd.DataFrame(results)

# --- END: per10.py -----------------------------------------------------------------


# %% [markdown]
# # %% [markdown]
# # ## 4. Merge core roster (01_merge_pipeline)
# #
# # We start by building a per-player, per-play roster table from the input tracking.
# # This mirrors `01_merge_pipeline.py` but uses the auto-detected `TRAIN` path.
# 

# %%
# %%
def load_concat(pattern: str, limit: int | None = None) -> pd.DataFrame:
    paths = sorted(TRAIN.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match train/{pattern}")
    if limit is not None:
        paths = paths[:limit]
    dfs = [pd.read_csv(p, low_memory=False) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def standardize_tracking_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common tracking column names to snake_case.
    """
    colmap = {
        "gameId": "game_id",
        "playId": "play_id",
        "nflId": "nfl_id",
        "frameId": "frame_id",
        "playerRole": "player_role",
        "playerSide": "player_side",
        "displayName": "player_name",
        "playerName": "player_name",
        "position": "player_position",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    df.columns = [c if c.islower() else c for c in df.columns]
    return df


def build_merged_core(limit_files: int | None = None) -> pd.DataFrame:
    inp = load_concat("input_2023_w*.csv", limit=limit_files) if list(TRAIN.glob("input_2023_w*.csv")) else load_concat("input_*.csv", limit=limit_files)
    inp = standardize_tracking_cols(inp)

    keep_cols = [
        c for c in [
            "game_id", "play_id", "nfl_id",
            "player_name", "player_position",
            "player_role", "player_side"
        ]
        if c in inp.columns
    ]
    if not keep_cols:
        raise ValueError("No expected roster columns found in input tracking.")

    roster = (
        inp[keep_cols]
        .drop_duplicates()
        .sort_values(["game_id", "play_id", "nfl_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    out_path = OUT / "merged_core.csv"
    roster.to_csv(out_path, index=False)
    print("merged_core.csv shape:", roster.shape)
    print("Wrote:", out_path.resolve())
    return roster


roster_df = build_merged_core(limit_files=2)
display(roster_df.head())

# %% [markdown]
# # %% [markdown]
# # ## 5. Pillar extraction and PER-10 traits (02_extract_pillars)
# #
# # Load input/output tracking and standardize columns.
# # Define the ball-in-air window for each play.
# # Compute PER-10 pillars from the ball-in-air frames.
# # Attach player side, write `per10_traits.csv`.
# # Compute WR/DB pillar features (anticipation ms, execution jitter, separation,
# #    coverage, innovation, eyes, reaction) and write them to `merged_pillars.csv`.

# %%
# %%
FPS = 10.0
MS_PER_FRAME = 1000.0 / FPS

COLMAP = {
    "gameId": "game_id",
    "playId": "play_id",
    "nflId": "nfl_id",
    "frameId": "frame_id",
    "playerRole": "player_role",
    "playerSide": "player_side",
}

REQUIRED = {
    "input":  ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "s", "o", "dir", "player_role"],
    "output": ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"],  # drop 'player_role' here
}



def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.rename(columns={k.lower(): v for k, v in COLMAP.items() if k.lower() in df.columns})
    return df


def load_concat_tracking(pattern: str, limit: int | None = None, kind: str = "input") -> pd.DataFrame:
    paths = sorted(TRAIN.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match train/{pattern}")
    if limit is not None:
        paths = paths[:limit]

    dfs = [pd.read_csv(p, low_memory=False) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = _standardize_cols(df)

    base_cols = set(REQUIRED[kind])
    missing = base_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {kind} files: {missing}. Have: {sorted(df.columns)}")

    if kind == "input" and "player_role" not in df.columns:
        raise KeyError("Input files must contain 'player_role'")

    return df


def first_move_frame(g):
    m = g[g["s"] > 0.5]
    return m["frame_id"].min() if len(m) else np.nan


def circ_std_deg(series):
    a = pd.Series(series).dropna().astype(float)
    if a.empty:
        return np.nan
    rad = np.deg2rad(a.to_numpy())
    C = np.mean(np.cos(rad))
    S = np.mean(np.sin(rad))
    R = np.hypot(C, S)
    if R <= 0:
        return np.nan
    std_rad = np.sqrt(max(0.0, -2.0 * np.log(R)))
    return np.rad2deg(std_rad)


def build_per10_and_pillars(limit_files: int | None = None):
    # 1) Load input/output tracking
    inp  = load_concat_tracking("input_2023_w*.csv",  limit=limit_files, kind="input") if list(TRAIN.glob("input_2023_w*.csv")) else load_concat_tracking("input_*.csv", limit=limit_files, kind="input")
    outp = load_concat_tracking("output_2023_w*.csv", limit=limit_files, kind="output") if list(TRAIN.glob("output_2023_w*.csv")) else load_concat_tracking("output_*.csv", limit=limit_files, kind="output")

    if "player_role" not in outp.columns:
        role_map = (
            inp[["game_id", "play_id", "nfl_id", "player_role"]]
            .drop_duplicates()
        )
        outp = outp.merge(role_map, on=["game_id", "play_id", "nfl_id"], how="left")

    # Map role → side for use in improv & separation
    inp["player_side"] = inp["player_role"].map(
        {"Targeted Receiver": "Offense", "Defensive Coverage": "Defense"}
    )

    # 2) Ball-in-air: anchors + window
    anchors = find_throw_and_arrival(inp, outp)
    inp_window = slice_ball_window(inp, anchors)

    # Add ball landing coordinates to the window for eyes/improv
    landing = compute_ball_landing(outp)
    window_df = inp_window.merge(
        landing,
        on=["game_id", "play_id"],
        how="left",
    )

    # Make sure ball_land_x / ball_land_y columns exist even if we couldn't
    # identify a landing point (keeps per10 logic unchanged and avoids KeyError).
    if "ball_land_x" not in window_df.columns:
        window_df["ball_land_x"] = np.nan
        window_df["ball_land_y"] = np.nan

    # 🔍 NEW: restrict to a subset of plays so the notebook finishes in time
    unique_plays = window_df[["game_id", "play_id"]].drop_duplicates().head(150)
    window_df = window_df.merge(
        unique_plays,
        on=["game_id", "play_id"],
        how="inner",
        suffixes=("", "_drop"),
    )

    # Show a sample ball-in-air window for one of the sampled plays
    sample_key = unique_plays.iloc[0].to_dict()
    sample_window = (
        window_df.query(
            "game_id == @sample_key['game_id'] and play_id == @sample_key['play_id']"
        )
        .sort_values("frame_id")
    )
    print("Sample ball-in-air window for game_id/play_id:", sample_key)
    display(sample_window[["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]].head(20))

    # 3) Compute PER-10 pillars (ball-in-air only)
    A_df      = compute_all_anticipation(window_df)
    S_df      = compute_all_separation(window_df)
    E_df      = compute_all_execution(window_df)
    Eyes_df   = compute_all_eyes(window_df)
    Innov_df  = compute_all_innovation(window_df)
    Improv_df = compute_all_improv(window_df)

    # Ensure each pillar dataframe has the expected columns,
    # even if it ends up empty for this subset of data.
    A_df = A_df.reindex(columns=["game_id", "play_id", "nfl_id", "A"])
    S_df = S_df.reindex(columns=["game_id", "play_id", "nfl_id", "S"])
    E_df = E_df.reindex(columns=["game_id", "play_id", "nfl_id", "E"])

    Eyes_df   = Eyes_df.reindex(columns=["game_id", "play_id", "nfl_id", "eyes_score"])
    Innov_df  = Innov_df.reindex(columns=["game_id", "play_id", "nfl_id", "innovation_score"])
    Improv_df = Improv_df.reindex(columns=["game_id", "play_id", "nfl_id", "improv_score"])

    per10_df = compute_all_per10_360(
        A_df, S_df, E_df, Eyes_df, Innov_df, Improv_df
    )

    # Quick Instinct flag (same as your script)
    per10_df["quick_instinct"] = (
        (per10_df["Eyes"] >= 8.0) &
        ((per10_df["Innovation"] >= 8.0) | (per10_df["Improv"] >= 7.0))
    ).astype(int)

    # Attach player_side from full input tracking
    meta_cols = (
        inp[["game_id", "play_id", "nfl_id", "player_role"]]
        .drop_duplicates()
        .assign(
            player_side=lambda df: df["player_role"].map(
                {"Targeted Receiver": "Offense", "Defensive Coverage": "Defense"}
            )
        )
        [["game_id", "play_id", "nfl_id", "player_side"]]
    )

    per10_traits = per10_df.merge(
        meta_cols,
        on=["game_id", "play_id", "nfl_id"],
        how="left",
    )

    per10_traits_path = OUT / "per10_traits.csv"
    per10_traits.to_csv(per10_traits_path, index=False)
    print("Saved PER-10 traits →", per10_traits_path.resolve(), "rows:", len(per10_traits))

    # ---- Legacy WR/DB pillar features for merged_pillars.csv ----

    # anchors for pre-throw logic
    t0 = (
        inp.groupby(["game_id", "play_id"])["frame_id"]
        .min()
        .reset_index()
        .rename(columns={"frame_id": "t0"})
    )
    t_throw = (
        inp.groupby(["game_id", "play_id"])["frame_id"]
        .max()
        .reset_index()
        .rename(columns={"frame_id": "t_throw"})
    )

    wr_in = inp[inp["player_role"].fillna("").str.contains("Targeted Receiver", case=False)].copy()
    db_in = inp[inp["player_role"].fillna("").str.contains("Defensive Coverage", case=False)].copy()

    # WR anticipation (ms)
    wr_move = (
        wr_in.groupby(["game_id", "play_id", "nfl_id"])
        .apply(first_move_frame)
        .reset_index(name="first_move")
    )
    wr_move = wr_move.merge(t0, on=["game_id", "play_id"], how="left")
    wr_move["anticipation_ms"] = (wr_move["first_move"] - wr_move["t0"]) * MS_PER_FRAME

    # WR execution jitter (std of dy)
    wr_path = wr_in[["game_id", "play_id", "nfl_id", "frame_id", "y"]].copy()
    wr_path["dy"] = wr_path.groupby(["game_id", "play_id", "nfl_id"])["y"].diff()
    wr_jitter = (
        wr_path.groupby(["game_id", "play_id", "nfl_id"], as_index=False)["dy"]
        .std()
        .rename(columns={"dy": "execution_jitter"})
    )

    # WR innovation: circular std of heading in last N frames pre-throw
    N = 10
    last_in = wr_in.merge(t_throw, on=["game_id", "play_id"], how="left")
    last_in = last_in[last_in["frame_id"] >= (last_in["t_throw"] - N)].copy()
    angle_col = "dir" if "dir" in last_in.columns else ("o" if "o" in last_in.columns else None)
    if angle_col is None:
        raise KeyError("Neither 'dir' nor 'o' found in input files.")
    wr_innov = (
        last_in.groupby(["game_id", "play_id", "nfl_id"])[angle_col]
        .apply(circ_std_deg)
        .reset_index(name="innovation_turn")
    )

    # earliest output frame per play
    t1 = (
        outp.groupby(["game_id", "play_id"])["frame_id"]
        .min()
        .reset_index()
        .rename(columns={"frame_id": "t1"})
    )
    o1 = outp.merge(t1, on=["game_id", "play_id"], how="left")
    o1 = o1[o1["frame_id"] == o1["t1"]].copy()

    wr_o = o1[o1["player_role"].fillna("").str.contains("Targeted Receiver", case=False)].copy()
    db_o = o1[o1["player_role"].fillna("").str.contains("Defensive Coverage", case=False)].copy()

    wr_o = wr_o.rename(columns={"nfl_id": "wr_id", "x": "x_wr", "y": "y_wr"})
    db_o = db_o.rename(columns={"nfl_id": "db_id", "x": "x_db", "y": "y_db"})
    pair = wr_o.merge(db_o, on=["game_id", "play_id"], how="inner")

    if not pair.empty:
        pair["dist"] = np.hypot(pair["x_wr"] - pair["x_db"], pair["y_wr"] - pair["y_db"])
        wr_sep = (
            pair.loc[
                pair.groupby(["game_id", "play_id", "wr_id"])["dist"].idxmin(),
                ["game_id", "play_id", "wr_id", "dist"],
            ]
            .rename(columns={"wr_id": "nfl_id", "dist": "separation_yds"})
        )
        db_cov = (
            pair.loc[
                pair.groupby(["game_id", "play_id"])["dist"].idxmin(),
                ["game_id", "play_id", "db_id", "dist"],
            ]
            .rename(columns={"db_id": "nfl_id", "dist": "coverage_yds"})
        )
    else:
        wr_sep = pd.DataFrame(columns=["game_id", "play_id", "nfl_id", "separation_yds"])
        db_cov = pd.DataFrame(columns=["game_id", "play_id", "nfl_id", "coverage_yds"])

    # DB reaction (ms)
    db_move = (
        db_in.groupby(["game_id", "play_id", "nfl_id"])
        .apply(first_move_frame)
        .reset_index(name="first_move")
    )
    db_move = db_move.merge(t0, on=["game_id", "play_id"], how="left")
    db_move["reaction_ms"] = (db_move["first_move"] - db_move["t0"]) * MS_PER_FRAME

    # Eyes for WRs (reuse Eyes_df)
    eyes = Eyes_df.rename(columns={"eyes_score": "eyes_score"})

    rows = []

    wr_feat = (
        wr_move[["game_id", "play_id", "nfl_id", "anticipation_ms"]]
        .merge(wr_jitter, on=["game_id", "play_id", "nfl_id"], how="left")
        .merge(wr_innov, on=["game_id", "play_id", "nfl_id"], how="left")
        .merge(wr_sep,   on=["game_id", "play_id", "nfl_id"], how="left")
        .merge(eyes,     on=["game_id", "play_id", "nfl_id"], how="left")
    )

    for _, r in wr_feat.iterrows():
        rows += [
            {
                "game_id": r.game_id,
                "play_id": r.play_id,
                "player_id": r.nfl_id,
                "side": "WR",
                "pillar": "anticipation",
                "raw_value": r.anticipation_ms,
                "units": "ms",
            },
            {
                "game_id": r.game_id,
                "play_id": r.play_id,
                "player_id": r.nfl_id,
                "side": "WR",
                "pillar": "execution",
                "raw_value": r.execution_jitter,
                "units": "yd",
            },
            {
                "game_id": r.game_id,
                "play_id": r.play_id,
                "player_id": r.nfl_id,
                "side": "WR",
                "pillar": "separation",
                "raw_value": r.separation_yds,
                "units": "yd",
            },
            {
                "game_id": r.game_id,
                "play_id": r.play_id,
                "player_id": r.nfl_id,
                "side": "WR",
                "pillar": "innovation",
                "raw_value": r.innovation_turn,
                "units": "deg",
            },
            {
                "game_id": r.game_id,
                "play_id": r.play_id,
                "player_id": r.nfl_id,
                "side": "WR",
                "pillar": "eyes",
                "raw_value": r.eyes_score,
                "units": "score",
            },
        ]

    for _, r in db_move.iterrows():
        rows.append(
            {
                "game_id": r.game_id,
                "play_id": r.play_id,
                "player_id": r.nfl_id,
                "side": "DB",
                "pillar": "reaction",
                "raw_value": r.reaction_ms,
                "units": "ms",
            }
        )
    for _, r in db_cov.iterrows():
        rows.append(
            {
                "game_id": r.game_id,
                "play_id": r.play_id,
                "player_id": r.nfl_id,
                "side": "DB",
                "pillar": "coverage",
                "raw_value": r.coverage_yds,
                "units": "yd",
            }
        )

    feats = pd.DataFrame(
        rows,
        columns=["game_id", "play_id", "player_id", "side", "pillar", "raw_value", "units"],
    )
    out_path = OUT / "merged_pillars.csv"
    feats.to_csv(out_path, index=False)
    print("Saved pillar features →", out_path.resolve(), "rows:", len(feats))

    return per10_traits, feats


per10_traits_df, merged_pillars_df = build_per10_and_pillars(limit_files=2)
display(per10_traits_df.head())
display(merged_pillars_df.head())


# %% [markdown]
# # %% [markdown]
# # ### 5.1 Quick look at PER-10 and pillar outputs

# %%
# %%
print("Pillar counts in merged_pillars.csv:")
print(merged_pillars_df["pillar"].value_counts())

print("\nPer-pillar summary of raw_value:")
display(merged_pillars_df.groupby("pillar")["raw_value"].describe())

plt.figure()
merged_pillars_df["raw_value"].hist(bins=20)
plt.title("Distribution of raw pillar values")
plt.xlabel("raw_value")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# # %% [markdown]
# # ## 6. Beta–binomial posterior (03_update_posterior)
# #
# # Interpret each WR/DB pillar value as a success/failure against a threshold.
# # Fit a beta–binomial model per player × side × pillar.
# # Convert the posterior mean to a 1–10 pillar score.
# # Combine pillars into an overall 0–100 rating.

# %%
# %%
PILLARS_CSV = OUT / "merged_pillars.csv"
CORE_CSV    = OUT / "merged_core.csv"

assert PILLARS_CSV.exists(), f"Missing {PILLARS_CSV}"
assert CORE_CSV.exists(),    f"Missing {CORE_CSV}"

feat_df = pd.read_csv(PILLARS_CSV, low_memory=False)

WR_WIN = {
    "anticipation": 120.0,  # ms, smaller is better
    "execution":    0.50,   # jitter in yds, smaller is better
    "separation":   1.50,   # yds, larger is better
    "innovation":   10.0,   # deg, larger is better
    # no 'eyes' here because eyes is a 1–10 score from PER-10 logic
}
DB_WIN = {
    "reaction": 140.0,      # ms, smaller is better
    "coverage": 1.25,       # yds, smaller is better
}

# eyes threshold on a 1–10 scale
EYES_THRESH_10 = 8.0

# Overall weights
WR_W = {"anticipation":0.30, "execution":0.20, "separation":0.20, "innovation":0.15, "eyes":0.15}
DB_W = {"coverage":0.50, "reaction":0.50}  # add "eyes":0.10 if you later include DB eyes

VALID_PILLARS = set(WR_W) | set(DB_W)


def pillar_success(side: str, pillar: str, v: float) -> int:
    if pd.isna(v):
        return 0
    if side == "WR":
        if pillar == "anticipation": return int(v <= WR_WIN["anticipation"])
        if pillar == "execution":    return int(v <= WR_WIN["execution"])
        if pillar == "separation":   return int(v >= WR_WIN["separation"])
        if pillar == "innovation":   return int(v >= WR_WIN["innovation"])
        if pillar == "eyes":         return int(v >= EYES_THRESH_10)   # Eyes is 1–10
    else:
        # DB side
        if pillar == "reaction":     return int(v <= DB_WIN["reaction"])
        if pillar == "coverage":     return int(v <= DB_WIN["coverage"])
        # If you later compute DB eyes as 1–10, you could add:
        # if pillar == "eyes":       return int(v >= EYES_THRESH_10)
    return 0


def beta_update(s: int, f: int, a0: float = 5.0, b0: float = 5.0):
    a = a0 + s
    b = b0 + f
    mean = a / (a + b) if (a + b) > 0 else np.nan
    lo, hi = (beta_dist.ppf([0.025, 0.975], a, b) if a > 0 and b > 0 else (np.nan, np.nan))
    return a, b, mean, lo, hi


def overall(side: str, means: dict) -> float:
    W = WR_W if side == "WR" else DB_W
    # only include pillars that exist; renormalize weights; scale to 0–100
    avail = {k: w for k, w in W.items() if k in means and pd.notna(means[k])}
    if not avail:
        return np.nan
    Z = sum(avail.values())
    return round(100.0 * sum((w / Z) * means[k] for k, w in avail.items()), 1)


def build_posteriors(feat_df: pd.DataFrame):
    df = feat_df.copy()

    # Handle either snake_case or legacy camelCase
    rename = {}
    if "playerId" in df.columns: rename["playerId"] = "player_id"
    if "gameId"   in df.columns: rename["gameId"]   = "game_id"
    if "playId"   in df.columns: rename["playId"]   = "play_id"
    if rename:
        df = df.rename(columns=rename)

    df["pillar"] = df["pillar"].astype(str)
    df = df[df["pillar"].isin(VALID_PILLARS)].copy()
    df["raw_value"] = pd.to_numeric(df["raw_value"], errors="coerce")

    # Success flag per event (player_id, side, pillar, rep)
    df["success"] = df.apply(
        lambda r: pillar_success(r["side"], r["pillar"], r["raw_value"]),
        axis=1,
    )

    # Aggregate successes/failures per player × side × pillar
    g = df.groupby(["player_id", "side", "pillar"])["success"].agg(["sum", "count"]).reset_index()
    g["fails"] = g["count"] - g["sum"]

    rows = []
    for _, r in g.iterrows():
        a, b, mean, lo, hi = beta_update(int(r["sum"]), int(r["fails"]))
        rows.append({
            "player_id": r["player_id"],
            "side": r["side"],
            "pillar": r["pillar"],
            "alpha": a,
            "beta": b,
            "mean": mean,
            "score_1_10": round(10 * mean) if pd.notna(mean) else np.nan,
            "ci_low": lo,
            "ci_high": hi,
        })

    pillars_post = pd.DataFrame(rows)

    # Player-level overall scores
    outs = []
    for (player_id, side), sub in pillars_post.groupby(["player_id", "side"]):
        means = {row["pillar"]: row["mean"] for _, row in sub.iterrows()}
        pill_vec = {f"mean_{p}": means.get(p, np.nan) for p in VALID_PILLARS}
        ovr = overall(side, means)
        per10_score = ovr / 10.0  # simple 0–10 overall proxy
        outs.append({
            "player_id": player_id,
            "side": side,
            **pill_vec,
            "overall_0_100": ovr,
            "per10": per10_score,
        })

    pillars_post.to_csv(OUT / "posterior" / "posterior_pillars.csv", index=False)

    overall_df = pd.DataFrame(outs)
    (OUT / "posterior").mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(OUT / "posterior" / "posterior_overall.csv", index=False)

    print("Wrote:", (OUT / "posterior" / "posterior_pillars.csv").resolve())
    print("Wrote:", (OUT / "posterior" / "posterior_overall.csv").resolve())
    return pillars_post, overall_df


POST_DIR = OUT / "posterior"
POST_DIR.mkdir(parents=True, exist_ok=True)

pillars_post_df, overall_post_df = build_posteriors(feat_df)
display(pillars_post_df.head())
display(overall_post_df.head())


# %% [markdown]
# # %% [markdown]
# # ### 6.1 A quick look at the posterior ratings

# %%
# %%
if "overall_0_100" in overall_post_df.columns:
    top_wr = (
        overall_post_df[overall_post_df["side"] == "WR"]
        .sort_values("overall_0_100", ascending=False)
        .head(10)
    )
    print("Top 10 WRs by overall 0–100 score:")
    display(top_wr)

    overall_post_df["overall_0_100"].hist(bins=20)
    plt.title("Distribution of overall 0–100 ratings")
    plt.xlabel("Score")
    plt.ylabel("Players")
    plt.show()

# %% [markdown]
# # %% [markdown]
# # ## 7. Final sanity check (for Restart & Run All)
# #
# # As a final check, we:
# # List the main CSVs written by the notebook.
# # Show the first few rows of each.
# #

# %%
# %%
print("Files in outputs/:")
for p in sorted(OUT.glob("**/*.csv")):
    print(" -", p.relative_to(OUT))

core = pd.read_csv(OUT / "merged_core.csv").head()
traits = pd.read_csv(OUT / "per10_traits.csv").head()
pill_post_head = pd.read_csv(OUT / "posterior" / "posterior_pillars.csv").head()
ovr_head = pd.read_csv(OUT / "posterior" / "posterior_overall.csv").head()

print("\nmerged_core.csv:")
display(core)

print("\nper10_traits.csv:")
display(traits)

print("\nposterior_pillars.csv:")
display(pill_post_head)

print("\nposterior_overall.csv:")
display(ovr_head)



