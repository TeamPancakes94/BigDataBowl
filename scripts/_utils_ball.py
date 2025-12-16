# scripts/_utils_ball.py

import pandas as pd

def find_throw_and_arrival(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-play anchors ['game_id','play_id','t_throw','t_arrival'].

    This approximates the ball-in-air window as the span of ball frames
    in the output tracking (e.g., NGS-style rows where team == 'football').

    If explicit ball rows are not available, we fall back to the earliest
    and latest frame in the output data for that play.
    """

    # Try to isolate ball rows (NGS: team == 'football')
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
        return anchors

    # Fallback: use the full range of output frames if no explicit ball rows
    anchors = (
        output_df.groupby(["game_id", "play_id"])["frame_id"]
        .agg(t_throw="min", t_arrival="max")
        .reset_index()
    )
    return anchors


def slice_ball_window(df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict tracking data to the ball-in-air window for each play:
    frame_id in [t_throw, t_arrival].
    """
    merged = df.merge(anchors, on=["game_id", "play_id"], how="inner")
    window = merged.query("frame_id >= t_throw and frame_id <= t_arrival").copy()
    return window
