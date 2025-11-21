import numpy as np
import pandas as pd

#=========================================================================================
# Computes Per-10 Score from given 5 pillar scores.
#=========================================================================================
def compute_per10_360(A, S, E, Eyes, Innovation, Improv):
    """
    Compute Ball IQ PER-10 score.
    Inputs must already be values from 1 to 10.
    """
    vals = np.array([A, S, E, Eyes, Innovation, Improv], dtype=float)
    per10 = np.round(vals.mean(), 0)
    return int(per10)

#=========================================================================================
# Outputs a DF with the Following Columns (basically a summary table) for a play id
# play_id,	nfl_player_id,	A,	S,	E,	Eyes,	Innovation,	 PER10
#=========================================================================================
def compute_all_per10_360(A_df, S_df, E_df, Eyes_df, Innovation_df, Improv_df):
    """
    Merge all pillar scores and compute PER-10 for every player-play.
    """

    # Merge all five pillar dataframes on (play_id, nfl_player_id)
    df = (
        A_df
        .merge(S_df, on=["play_id", "nfl_player_id"], how="outer")
        .merge(E_df, on=["play_id", "nfl_player_id"], how="outer")
        .merge(Eyes_df.rename(columns={"eyes_score": "Eyes"}), 
               on=["play_id", "nfl_player_id"], how="outer")
        .merge(Innovation_df.rename(columns={"innovation_score": "Innovation"}), 
               on=["play_id", "nfl_player_id"], how="outer")
        .merge(Improv_df.rename(columns={"improv_score": "Improv"}),
               on=["play_id", "nfl_player_id"], how="outer")
    )

    # Compute PER-10 for each row
    df["PER10_360"] = df.apply(
        lambda row: compute_per10_360(
            row["A"], 
            row["S"], 
            row["E"], 
            row["Eyes"], 
            row["Innovation"],
            row["Improv"]
        ),
        axis=1
    )

    return df

#=========================================================================================
# How Anticipation Was Calculated:
# Look at the ball position (ball_land_x, ball_land_y).
# Use player movement vectors to detect when they “turn toward” the destination.
# How many frames after release until the player’s movement vector points toward (ball_land_x, ball_land_y)
#=========================================================================================
def score_A(anticipation_frames):
    """
    anticipation_frames = how many frames after release the player reacts.
    """
    if anticipation_frames <= 1: return 10
    if anticipation_frames <= 3: return 8
    if anticipation_frames <= 5: return 6
    if anticipation_frames <= 7: return 4
    return 2



# -----------------------------------------------
# Main function: compute anticipation for one player in one play
# -----------------------------------------------
def compute_anticipation_for_group(g):
    """
    g = dataframe containing only ONE (play_id, nfl_id)
    Must have columns: x, y, frame_id, ball_land_x, ball_land_y
    """

    g = g.sort_values("frame_id").copy()

    # Movement vectors
    g["dx"] = g["x"].diff()
    g["dy"] = g["y"].diff()

    # Vector toward ball landing point
    g["bx"] = g["ball_land_x"] - g["x"]
    g["by"] = g["ball_land_y"] - g["y"]

    # Alignment = movement toward ball
    g["alignment"] = g["dx"] * g["bx"] + g["dy"] * g["by"]

    # Normalize per-frame values
    g["alignment_norm"] = (g["alignment"] - g["alignment"].min()) / \
                          (g["alignment"].max() - g["alignment"].min() + 1e-6)

    # Reaction frame = first frame where alignment_norm > threshold
    reaction_frame = g.index[g["alignment_norm"] > 0.25][0] \
                     if (g["alignment_norm"] > 0.25).any() else None

    if reaction_frame is None:
        return 2  # no reaction detected = worst score

    # frame delay from first frame of window
    frame_delay = reaction_frame - g.index[0]

    return score_A(frame_delay)


# -----------------------------------------------
# Apply to whole dataset
# -----------------------------------------------
def compute_all_anticipation(df_input):
    """
    Returns dataframe with: play_id, nfl_id, A_score
    """
    results = []

    for (play_id, nfl_id), g in df_input.groupby(["play_id", "nfl_id"]):
        A = compute_anticipation_for_group(g)
        results.append([play_id, nfl_id, A])

    return pd.DataFrame(results, columns=["play_id", "nfl_id", "A"])



#=========================================================================================
# How Separation was Calculated:
# Distance between WR and CB per frame
# WRs Create Separation and CBs Deny Separation
# Select WR with smallest distance to ball at first frame.
# Pick the closest CB to the WR
# Compute separation over every frame
#=========================================================================================
def score_S(distance_start, distance_end):
    """
    Positive = defender closes separation.
    Negative = loses separation.
    """
    diff = distance_start - distance_end

    if diff >= 1.5: return 10
    if diff >= 1.0: return 8
    if diff >= 0.5: return 6
    if diff >= 0:   return 5
    return 3  # lost separation

# -----------------------------------------------------
# Compute separation for one play (WR vs CB)
# -----------------------------------------------------
def compute_separation_for_play(g):
    """
    g = dataframe for a single play_id, containing WR and CB players.
    """

    # Identify WR and CB rows
    wr_df = g[g["player_position"] == "WR"].copy()
    cb_df = g[g["player_position"] == "CB"].copy()

    if wr_df.empty or cb_df.empty:
        return None  # can't compute separation

    # For this simplified model, assume:
    # - first WR is the target
    wr_id = wr_df["nfl_id"].iloc[0]

    wr = wr_df[wr_df["nfl_id"] == wr_id].sort_values("frame_id")

    separations = []

    # Loop by frame
    for f, wr_row in wr.iterrows():

        cb_df_same_frame = cb_df[cb_df["frame_id"] == wr_row["frame_id"]]
        if cb_df_same_frame.empty:
            continue

        # distance to each CB, keep the minimum (primary defender)
        dists = np.sqrt((cb_df_same_frame["x"] - wr_row["x"])**2 +
                        (cb_df_same_frame["y"] - wr_row["y"])**2)

        separations.append(dists.min())

    if len(separations) == 0:
        return None

    avg_sep = np.mean(separations)

    return score_S(avg_sep)


# -----------------------------------------------------
# Apply to whole dataset
# -----------------------------------------------------
def compute_all_separation(df_input):
    """
    Returns dataframe with: play_id, S score
    Each play produces one Separation score.
    """

    results = []

    for play_id, g in df_input.groupby("play_id"):
        S = compute_separation_for_play(g)
        if S is not None:
            results.append([play_id, S])

    return pd.DataFrame(results, columns=["play_id", "S"])




#=========================================================================================
# How Execution was Calculated:
# Good execution indicators:
#   Smooth directional changes (low jerk)
#   Stable speed (not panicked accelerations)
#   Body orientation aligned with intended path
#   Low “wasted movement”
# 3 Metrics to Calculate Execution:
#   Angle Smoothness (Movement Jerk): Lower jerk = better technique
#   Body Control (Orientation vs. Movement Alignment):
#       If a player’s orientation (o) and movement direction (dir) differ too much, he is off-balance.
#   Leverage Stability (Path relative to defender): 
#       For WR: Maintaining vertical stack = good | Being forced off path = bad
#       For CB: Staying square / maintaining leverage = good | Getting turned around = bad
# WRs Create Separation and CBs Deny Separation
#=========================================================================================
def score_E(orientation_std, speed_std):
    """
    Lower variability → better technique.
    """
    chaos = orientation_std + speed_std

    if chaos < 0.5: return 10
    if chaos < 1.0: return 8
    if chaos < 1.5: return 6
    if chaos < 2.0: return 5
    return 3

# ------------------------------------------------------
# Helper: angle difference accounting for wrap-around
# ------------------------------------------------------
def angle_diff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


# ------------------------------------------------------
# Compute E for a single (play_id, nfl_id)
# ------------------------------------------------------
def compute_execution_for_group(g):

    g = g.sort_values("frame_id").copy()

    # Must have at least 3 frames to measure smoothness
    if len(g) < 3:
        return 5   # neutral score
    
    # ------------ E1: Movement jerk (direction smoothness) ------------
    dir_change = [
        angle_diff(g["dir"].iloc[i], g["dir"].iloc[i-1])
        for i in range(1, len(g))
    ]
    jerk = np.mean(dir_change)
    E1 = 1 / (1 + jerk)  # smaller jerk → bigger E1

    # ------------ E2: Orientation vs Movement alignment ------------
    misalign = [
        angle_diff(g["dir"].iloc[i], g["o"].iloc[i])
        for i in range(len(g))
    ]
    misalign_norm = np.mean(misalign) / 180  # normalize 0–1
    E2 = 1 - misalign_norm  # bigger is better

    # ------------ E3: Path smoothness (vector angle change) ----------
    dx = g["x"].diff().fillna(0)
    dy = g["y"].diff().fillna(0)

    path_angles = []
    for i in range(2, len(g)):
        v1 = np.array([dx.iloc[i-1], dy.iloc[i-1]])
        v2 = np.array([dx.iloc[i], dy.iloc[i]])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue
        # angle between vectors
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosang = np.clip(cosang, -1, 1)
        angle = np.degrees(np.arccos(cosang))
        path_angles.append(angle)

    if len(path_angles) == 0:
        E3 = 0.5
    else:
        path_smoothness = 1 - (np.mean(np.abs(path_angles)) / 180)
        E3 = path_smoothness

    # ------------ Combine E1, E2, E3 into PER-10 score ------------
    raw_E = np.mean([E1, E2, E3])  # 0–1
    E_score = int(np.round(raw_E * 10))

    return E_score


# ------------------------------------------------------
# Apply to the entire dataset
# ------------------------------------------------------
def compute_all_execution(df_input):
    """
    Returns per-player-per-play Execution score.
    """
    results = []

    for (play_id, nfl_id), g in df_input.groupby(["play_id", "nfl_id"]):
        E = compute_execution_for_group(g)
        results.append([play_id, nfl_id, E])

    return pd.DataFrame(results, columns=["play_id", "nfl_id", "E"])



#=========================================================================================
# How Eyes Was Calculated:
# How early and how consistently did the player orient toward the ball’s landing spot?
# Angular Alignment (50%):
#   How close the player’s head/torso orientation (o) is to the direction of the ball landing spot.
# Tracking Stability (30%):
#   Do they keep their head consistently pointed toward the ball?
# Tracking Reaction Time (20%):
#   How many frames after release until the player orients within 20° of the ball trajectory.
#=========================================================================================
def score_Eyes(lockon_frame):
    """
    lockon_frame = frames after release until head is oriented toward ball.
    """
    if lockon_frame <= 1: return 10
    if lockon_frame <= 3: return 8
    if lockon_frame <= 5: return 6
    if lockon_frame <= 7: return 4
    return 2

# ------------------------------------------------------
# Compute Eyes for one player in one play
# ------------------------------------------------------
def compute_eyes_score(play_df, landing_point, player_id):
    """
    Computes the Eyes (Ball Tracking / Vision) score for a single player in a single play.
    
    Parameters
    ----------
    play_df : pd.DataFrame
        Tracking rows for the specific play (ball-in-air frames only).
    landing_point : tuple(float, float)
        (ball_x, ball_y) landing coordinates.
    player_id : int
        NFL player id for the target defender/receiver.
    
    Returns
    -------
    float
        Eyes score on a 0–10 scale.
    """

    px, py = landing_point

    # filter frames for this player
    df = play_df[play_df['nfl_player_id'] == player_id].copy()
    if df.empty:
        return np.nan

    # ---- 1. Compute angular error between head orientation and ball (degrees)
    dx = px - df['x']
    dy = py - df['y']

    # angle from player → ball (global angle)
    ball_angle = np.degrees(np.arctan2(dy, dx)) % 360

    # orientation is already in degrees
    orientation = df['o'] % 360

    # smallest absolute angular difference (0 to 180)
    angle_error = np.abs(((orientation - ball_angle + 180) % 360) - 180)

    # ---- 2. Tracking stability (low variance = good tracking)
    stability_penalty = np.std(angle_error)

    # ---- 3. How early did they orient toward the ball?
    # (frames until error < 20°)
    threshold = 20
    frames_until_track = np.argmax(angle_error < threshold) if any(angle_error < threshold) else len(angle_error)

    # normalize components to 0–10
    # lower = better
    angle_component = np.clip(10 - (np.mean(angle_error) / 18), 0, 10)   # 180° → 0,   0° → 10
    stability_component = np.clip(10 - (stability_penalty / 10), 0, 10)  # high variance → lower score
    reaction_component = np.clip(10 - (frames_until_track * 1.5), 0, 10)

    # weighted eyes metric
    eyes_score = (
        0.5 * angle_component +
        0.3 * stability_component +
        0.2 * reaction_component
    )

    return round(float(eyes_score), 2)


# ------------------------------------------------------
# Apply Eyes scoring to the entire dataset
# ------------------------------------------------------
def compute_all_eyes(df, landing_points):
    """
    Computes Eyes score for all players in all plays.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking dataframe (ball-in-air frames only).
    landing_points : dict
        Mapping play_id -> (landing_x, landing_y) tuple.

    Returns
    -------
    pd.DataFrame
        Columns: play_id, nfl_player_id, eyes_score
    """

    results = []

    # Loop through each play
    for play_id, play_df in df.groupby("play_id"):

        # Must have landing point info
        if play_id not in landing_points:
            continue

        landing_point = landing_points[play_id]

        # Loop through each player in the play
        for player_id in play_df["nfl_player_id"].unique():

            score = compute_eyes_score(
                play_df=play_df,
                landing_point=landing_point,
                player_id=player_id
            )

            results.append({
                "play_id": play_id,
                "nfl_player_id": player_id,
                "eyes_score": score
            })

    return pd.DataFrame(results)



#=========================================================================================
# How Innovation Was Calculated:
# Did the player make an unexpected, effective adjustment compared to their typical movement pattern?
# Sudden directional changes (cuts, pivots) during the ball-in-air window
# That reduce separation (defender) or improve positioning (receiver)
# Compared to what a linear/extrapolated movement model would predict
# Measuers:
# Directional Creativity (cuts & pivots): 
#   Large directional changes = reacting creatively mid-rep.
# Nonlinear Movement (deviation from expected path)
#   If the player deviates from a straight predicted movement, this is "innovation".
#Effectiveness of adjustment
#   Did the action improve the outcome? Receiver: moved closer to catch point, Defender: reduced separation
#=========================================================================================
def score_Innovation(innovation_metric):
    """
    innovation_metric: ratio of unexpected adjustments.
    >1 = creative, <1 = passive.
    """
    if innovation_metric >= 1.5: return 10
    if innovation_metric >= 1.2: return 8
    if innovation_metric >= 1.0: return 6
    if innovation_metric >= 0.8: return 5
    return 3


# ------------------------------------------------------
# Compute Innovation for one player in one play
# ------------------------------------------------------
def compute_innovation_score(play_df, player_id, role="receiver"):
    """
    Computes Innovation score (0–10) for one player in one play.
    
    Innovation = Did the player make an unexpected, effective movement adjustment?
    
    Parameters
    ----------
    play_df : pd.DataFrame
        Tracking rows for a specific play (ball-in-air frames only).
    player_id : int
        NFL ID of player being evaluated.
    role : str
        "receiver" or "defender" to determine how to judge effectiveness.
        
    Returns
    -------
    float
        Innovation score on 0–10 scale.
    """

    df = play_df[play_df["nfl_player_id"] == player_id].copy()
    if df.empty or len(df) < 3:
        return np.nan

    # ---- 1. Compute frame-to-frame direction changes (in degrees)
    dx = df["x"].diff()
    dy = df["y"].diff()

    directions = np.degrees(np.arctan2(dy, dx)) % 360
    direction_change = np.abs((directions.diff() + 180) % 360 - 180)

    # Big changes imply "creative adjustments"
    mean_cut_angle = np.mean(direction_change[1:])  # skip first NaN

    # ---- 2. Compute expected linear movement (no adjustments)
    # Fit linear model in x and y vs frames
    frames = df["frame_id"].values
    coef_x = np.polyfit(frames, df["x"], 1)
    coef_y = np.polyfit(frames, df["y"], 1)

    pred_x = np.polyval(coef_x, frames)
    pred_y = np.polyval(coef_y, frames)

    # Deviation from predicted path
    deviation = np.sqrt((pred_x - df["x"])**2 + (pred_y - df["y"])**2)
    mean_deviation = np.mean(deviation)

    # ---- 3. Effectiveness of adjustment
    # Receiver: getting closer to landing point
    # Defender: getting closer to receiver
    effectiveness = 0

    if role == "receiver":
        if "landing_x" in df and "landing_y" in df:
            final_dist = np.sqrt((df["x"].iloc[-1] - df["landing_x"].iloc[-1])**2 +
                                 (df["y"].iloc[-1] - df["landing_y"].iloc[-1])**2)
            initial_dist = np.sqrt((df["x"].iloc[0] - df["landing_x"].iloc[0])**2 +
                                   (df["y"].iloc[0] - df["landing_y"].iloc[0])**2)
            effectiveness = initial_dist - final_dist  # improvement
    else:  # defender
        if "target_separation" in df:
            effectiveness = df["target_separation"].iloc[0] - df["target_separation"].iloc[-1]

    # ---- 4. Normalize components to 0–10
    cut_component = np.clip(mean_cut_angle / 15, 0, 10)  # lots of cuts = creative
    deviation_component = np.clip(mean_deviation / 0.8, 0, 10)  # moved creatively vs straight line
    effectiveness_component = np.clip(effectiveness, 0, 10)

    # Weighted innovation score
    innovation = (
        0.4 * cut_component +
        0.3 * deviation_component +
        0.3 * effectiveness_component
    )

    return round(float(innovation), 2)

# ------------------------------------------------------
# Compute Innovation for the entire dataset
# ------------------------------------------------------
def compute_all_innovation(df, role_map=None):
    """
    Computes Innovation scores for all players in all plays.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking dataset (should already be filtered to ball-in-air frames).
    role_map : dict, optional
        Dictionary mapping player_id -> role ("receiver" or "defender").
        If None, defaults all players to "receiver".

    Returns
    -------
    pd.DataFrame
        Columns: play_id, nfl_player_id, innovation_score
    """

    results = []

    # Loop through each play
    for play_id, play_df in df.groupby("play_id"):

        # Loop through each player in that play
        for player_id in play_df["nfl_player_id"].unique():

            # Get role for player
            role = "receiver"
            if role_map is not None and player_id in role_map:
                role = role_map[player_id]

            # Compute Innovation for this player in this play
            score = compute_innovation_score(play_df, player_id, role)

            results.append({
                "play_id": play_id,
                "nfl_player_id": player_id,
                "innovation_score": score
            })

    return pd.DataFrame(results)


#=========================================================================================
# How Improv Was Calculated:
# Did the player make an unexpected, effective adjustment compared to their typical movement pattern?
# Disruption Detection - Did the player get forced off their expected path?
#   We measure how often the player is unexpectedly displaced: sudden deceleration, sudden direction flip
# Recovery Speed - Once disrupted, how quickly did the player recover?
#   Measure: TIME between disruption moment and re-stabilization
# Improvised Outcome Gain - Did the reaction actually improve the play?
#   Receiver: moved closer to ball | Defender: reduced separation
#=========================================================================================
def compute_improv_score(play_df, player_id, role="receiver"):
    """
    Computes Improv Index (0–10) for one player in one play.

    Improv = reactive recovery and adaptation AFTER play breaks structure.
    """

    df = play_df[play_df["nfl_player_id"] == player_id].copy()
    if df.empty or len(df) < 5:
        return np.nan

    # 1. Detect disruption (D)
    # sudden speed changes
    speed = df["s"].values
    speed_diff = np.abs(np.diff(speed))

    # sudden direction changes
    directions = np.degrees(np.arctan2(df["y"].diff(), df["x"].diff())) % 360
    dir_diff = np.abs((np.diff(directions) + 180) % 360 - 180)

    # measure disruption magnitude
    disruption_raw = np.mean(speed_diff) + (np.mean(dir_diff) / 2)

    # normalize to 0–10
    D = np.clip(disruption_raw / 5, 0, 10)  # 5 is typical max in tracking data

    # 2. Recovery Speed (R)
    # recovery = how fast direction variance decreases after the largest spike
    if len(dir_diff) < 3:
        return 5.0

    disruption_frame = np.argmax(dir_diff)  # moment of chaos

    post = dir_diff[disruption_frame:]
    if len(post) < 2:
        R = 5
    else:
        # how fast jitter declines
        recover_time = np.argmax(post < 10) if np.any(post < 10) else len(post)
        R = np.clip(10 - recover_time * 1.5, 0, 10)

    # 3. Outcome Gain (G)
    G = 0

    if role == "receiver":
        # improvement in distance to landing point
        if "landing_x" in df and "landing_y" in df:
            lx = df["landing_x"].iloc[0]
            ly = df["landing_y"].iloc[0]

            initial_dist = np.sqrt((df["x"].iloc[0] - lx)**2 +
                                   (df["y"].iloc[0] - ly)**2)
            final_dist = np.sqrt((df["x"].iloc[-1] - lx)**2 +
                                 (df["y"].iloc[-1] - ly)**2)
            G = np.clip(initial_dist - final_dist, 0, 10)

    else:  # defender
        if "target_separation" in df:
            sep0 = df["target_separation"].iloc[0]
            sep1 = df["target_separation"].iloc[-1]
            G = np.clip(sep0 - sep1, 0, 10)

    # Combine
    improv = 0.4 * D + 0.3 * R + 0.3 * G
    return round(float(improv), 2)

# ------------------------------------------------------
# Compute Innovation for the entire dataset
# ------------------------------------------------------
def compute_all_improv(df, role_map=None):
    """
    Computes Improv Index (I) for all players in all plays.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking dataset (already filtered to ball-in-air frames).
    role_map : dict, optional
        Mapping player_id -> role ("receiver" or "defender").
        If None, defaults all players to "receiver".

    Returns
    -------
    pd.DataFrame
        Columns: play_id, nfl_player_id, improv_score
    """

    results = []

    # Loop through each play
    for play_id, play_df in df.groupby("play_id"):

        # Loop through each player in the play
        for player_id in play_df["nfl_player_id"].unique():

            # Assign role
            role = "receiver"
            if role_map is not None and player_id in role_map:
                role = role_map[player_id]

            # Compute improv score for that player in that play
            score = compute_improv_score(play_df, player_id, role)

            results.append({
                "play_id": play_id,
                "nfl_player_id": player_id,
                "improv_score": score
            })

    return pd.DataFrame(results)