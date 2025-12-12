# scripts/per10.py

import numpy as np
import pandas as pd

# =============================================================================
# PER-10 (5-pillars) and PER-10 360 (6-pillars)
# =============================================================================

def compute_per10_basic(A, S, E, Eyes, Innovation):
    """
    5-pillar PER-10:
    PER-10 = average(A, S, E, Eyes, Innovation) on a 1–10 scale.
    """
    vals = np.array([A, S, E, Eyes, Innovation], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.round(np.nanmean(vals), 0))


def compute_per10_360(A, S, E, Eyes, Innovation, Improv):
    """
    6-pillar PER-10 360:
    PER-10 360 = average(A, S, E, Eyes, Innovation, Improv) on a 1–10 scale.
    """
    vals = np.array([A, S, E, Eyes, Innovation, Improv], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    per10_360 = np.nanmean(vals)
    return float(np.round(per10_360, 0))


def compute_all_per10_360(A_df, S_df, E_df, Eyes_df, Innovation_df, Improv_df):
    """
    Merge all pillar scores and compute PER-10 and PER-10 360 for every player-play.

    Inputs must contain:
      - game_id, play_id, nfl_id, pillar_value
    """

    df = (
        A_df
        .merge(S_df,      on=["game_id", "play_id", "nfl_id"], how="outer")
        .merge(E_df,      on=["game_id", "play_id", "nfl_id"], how="outer")
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

    # Compute both PER-10 variants
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

# =============================================================================
# Anticipation (A)
# =============================================================================

def score_A(anticipation_frames):
    if anticipation_frames <= 1: return 10
    if anticipation_frames <= 3: return 8
    if anticipation_frames <= 5: return 6
    if anticipation_frames <= 7: return 4
    return 2


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
        (g["alignment"] - g["alignment"].min()) /
        (g["alignment"].max() - g["alignment"].min() + 1e-6)
    )

    mask = g["alignment_norm"] > 0.25
    if not mask.any():
        return 2

    reaction_idx = g.index[mask][0]
    frame_delay = reaction_idx - g.index[0]

    return score_A(frame_delay)


def compute_all_anticipation(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: game_id, play_id, nfl_id, A
    """
    results = []
    for (game_id, play_id, nfl_id), g in df_input.groupby(["game_id", "play_id", "nfl_id"]):
        A = compute_anticipation_for_group(g)
        results.append([game_id, play_id, nfl_id, A])

    return pd.DataFrame(results, columns=["game_id", "play_id", "nfl_id", "A"])

# =============================================================================
# Separation (S)
# =============================================================================

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
        if delta_sep <= -3:   return 10
        if delta_sep <= -2:   return 8
        if delta_sep <= -1:   return 6
        if delta_sep <= -0.5: return 5
        return 3


def compute_player_separation(play_df: pd.DataFrame, player_id) -> float | None:
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

    score = score_separation(delta_sep, role)
    return round(float(score), 2)


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

# =============================================================================
# Execution (E)
# =============================================================================

def angle_diff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def compute_execution_for_group(g: pd.DataFrame) -> int:
    g = g.sort_values("frame_id").copy()
    if len(g) < 3:
        return 5  # neutral

    # Direction smoothness
    dir_change = [
        angle_diff(g["dir"].iloc[i], g["dir"].iloc[i - 1])
        for i in range(1, len(g))
    ]
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
    path_angles = []
    for i in range(2, len(g)):
        v1 = np.array([dx.iloc[i - 1], dy.iloc[i - 1]])
        v2 = np.array([dx.iloc[i], dy.iloc[i]])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosang = np.clip(cosang, -1, 1)
        angle = np.degrees(np.arccos(cosang))
        path_angles.append(angle)

    if not path_angles:
        E3 = 0.5
    else:
        path_smoothness = 1 - (np.mean(np.abs(path_angles)) / 180)
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

# =============================================================================
# Eyes (tracking / vision)
# =============================================================================

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

    angle_error = np.abs(((orientation - ball_angle + 180) % 360) - 180)

    stability_penalty = np.std(angle_error)

    threshold = 20
    if (angle_error < threshold).any():
        frames_until_track = np.argmax(angle_error < threshold)
    else:
        frames_until_track = len(angle_error)

    angle_component = np.clip(10 - (np.mean(angle_error) / 18), 0, 10)
    stability_component = np.clip(10 - (stability_penalty / 10), 0, 10)
    reaction_component = np.clip(10 - (frames_until_track * 1.5), 0, 10)

    eyes_score = (
        0.5 * angle_component +
        0.3 * stability_component +
        0.2 * reaction_component
    )

    return round(float(eyes_score), 2)


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

# =============================================================================
# Innovation
# =============================================================================

def score_Innovation(innovation_metric):
    if innovation_metric >= 1.5: return 10
    if innovation_metric >= 1.2: return 8
    if innovation_metric >= 1.0: return 6
    if innovation_metric >= 0.8: return 5
    return 3


def compute_innovation_score(play_df: pd.DataFrame, player_id, role: str) -> float:
    df = play_df[play_df["nfl_id"] == player_id].copy()
    if df.empty or len(df) < 3:
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

    effectiveness = 0.0
    if role == "Offense":
        if "ball_land_x" in df.columns and "ball_land_y" in df.columns:
            final_dist = np.sqrt(
                (df["x"].iloc[-1] - df["ball_land_x"].iloc[-1]) ** 2 +
                (df["y"].iloc[-1] - df["ball_land_y"].iloc[-1]) ** 2
            )
            initial_dist = np.sqrt(
                (df["x"].iloc[0] - df["ball_land_x"].iloc[0]) ** 2 +
                (df["y"].iloc[0] - df["ball_land_y"].iloc[0]) ** 2
            )
            effectiveness = initial_dist - final_dist
    else:
        if "target_separation" in df.columns:
            effectiveness = df["target_separation"].iloc[0] - df["target_separation"].iloc[-1]

    cut_component = np.clip(mean_cut_angle / 15, 0, 10)
    deviation_component = np.clip(mean_deviation / 0.8, 0, 10)
    effectiveness_component = np.clip(effectiveness, 0, 10)

    innovation = (
        0.4 * cut_component +
        0.3 * deviation_component +
        0.3 * effectiveness_component
    )

    return round(float(innovation), 2)


def compute_all_innovation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: game_id, play_id, nfl_id, innovation_score
    """
    results = []
    for (game_id, play_id), play_df in df.groupby(["game_id", "play_id"]):
        for player_id in play_df["nfl_id"].unique():
            player_rows = play_df[play_df["nfl_id"] == player_id]
            if player_rows.empty:
                continue
            role = player_rows["player_side"].iloc[0]
            score = compute_innovation_score(play_df, player_id, role)
            results.append({
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": player_id,
                "innovation_score": score,
            })

    return pd.DataFrame(results)

# =============================================================================
# Improv Index (I)
# =============================================================================

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

    if len(dir_diff) < 3:
        return 5.0

    disruption_frame = int(np.argmax(dir_diff))
    post = dir_diff[disruption_frame:]
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
