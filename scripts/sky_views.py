# scripts/sky_views.py
# Movement & spatial view + Play Deep Dive 

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
CORE_CSV = OUT_DIR / "merged_core.csv"
COORDS_CSV = OUT_DIR / "coordinates.csv"

# Field constants + helper for auto-zoom ---------------------------------------

FIELD_LENGTH = 120.0   # full field in yards (including end zones)
FIELD_WIDTH = 53.3     # standard NFL width in yards

def _auto_bounds(xs, ys, x_margin: float = 5.0, y_margin: float = 5.0):
    """
    Compute a zoom window around all points in xs / ys, with some margin
    and a minimum window size so the chart never collapses to a tiny sliver.
    """
    xs = np.array(xs)
    ys = np.array(ys)

    xmin, xmax = xs.min() - x_margin, xs.max() + x_margin
    ymin, ymax = ys.min() - y_margin, ys.max() + y_margin

    # clamp to field
    xmin = max(0.0, xmin)
    xmax = min(FIELD_LENGTH, xmax)
    ymin = max(0.0, ymin)
    ymax = min(FIELD_WIDTH, ymax)

    # enforce a minimum width/height
    if xmax - xmin < 15:
        cx = 0.5 * (xmin + xmax)
        xmin = max(0.0, cx - 7.5)
        xmax = min(FIELD_LENGTH, cx + 7.5)

    if ymax - ymin < 10:
        cy = 0.5 * (ymin + ymax)
        ymin = max(0.0, cy - 5.0)
        ymax = min(FIELD_WIDTH, cy + 5.0)

    return xmin, xmax, ymin, ymax


# helpers -------------------

def plot_separation_timeline(frames_wr: pd.DataFrame,
                             frames_db: pd.DataFrame,
                             target_frame: int | None = None):
    """
    Build a beautiful separation-over-time chart for a WR–DB pair.

    frames_wr / frames_db: data for a *single* play and player each.
      Expected columns (rename if yours differ):
        - frame_id  : int
        - x, y      : coordinates in yards

    target_frame: optional frame_id where the ball arrives / is targeted.
    """

    # 1) Align WR & DB by frame and compute separation
    wr = frames_wr[["frame_id", "x", "y"]].rename(
        columns={"x": "wr_x", "y": "wr_y"}
    )
    db = frames_db[["frame_id", "x", "y"]].rename(
        columns={"x": "db_x", "y": "db_y"}
    )

    pair = wr.merge(db, on="frame_id", how="inner").sort_values("frame_id")
    if pair.empty:
        return None  # caller can handle with a warning

    pair["separation"] = np.sqrt(
        (pair["wr_x"] - pair["db_x"]) ** 2 + (pair["wr_y"] - pair["db_y"]) ** 2
    )

    # 2) Basic line chart
    fig = px.line(
        pair,
        x="frame_id",
        y="separation",
        labels={
            "frame_id": "Frame (snap → throw)",
            "separation": "Separation (yards)",
        },
    )

    # 3) NFL open band (e.g. 2–3 yards)
    NFL_OPEN_LOW = 2.0
    NFL_OPEN_HIGH = 3.0
    FRAMES_PER_SECOND = 10.0

    open_low = NFL_OPEN_LOW
    open_high = NFL_OPEN_HIGH

    fig.add_trace(
        go.Scatter(
            x=list(pair["frame_id"]) + list(pair["frame_id"][::-1]),
            y=[open_low] * len(pair) + [open_high] * len(pair),
            fill="toself",
            fillcolor="rgba(56,189,248,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="NFL open window (2–3 yds)",
        )
    )

    # 4) Ball arrival marker (if known)
    if target_frame is not None and target_frame in pair["frame_id"].values:
        sep_at_target = float(
            pair.loc[pair["frame_id"] == target_frame, "separation"].iloc[0]
        )
        fig.add_vline(
            x=target_frame,
            line_dash="dash",
            line_width=1.5,
            line_color="#facc15",
            annotation_text=f"Ball arrives\n{sep_at_target:.2f} yds",
            annotation_position="top right",
        )

    # 5) Styling — match your dark navy theme
    fig.update_traces(
        line=dict(width=3, color="#38bdf8"),
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        margin=dict(l=40, r=30, t=40, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# Data loader for coordinates + roster (Movement & Spatial tab) -------------------------

@st.cache_data(show_spinner=True)
def load_coords() -> pd.DataFrame:
    """
    Load coordinates.csv and attach roster info (names/positions/side).
    """
    if not COORDS_CSV.exists():
        raise FileNotFoundError(
            f"coordinates.csv not found at {COORDS_CSV}. "
            "Run pose_ball_proxy.py (or move the file) before using this tab."
        )
    if not CORE_CSV.exists():
        raise FileNotFoundError(
            f"merged_core.csv not found at {CORE_CSV}. "
            "Run 01_merge_pipeline.py first."
        )

    coords = pd.read_csv(COORDS_CSV)
    core = pd.read_csv(CORE_CSV)

    # normalize columns
    for df in (coords, core):
        df.columns = [c.lower() for c in df.columns]

    # harmonize id col
    if "nfl_id" in coords.columns:
        coords = coords.rename(columns={"nfl_id": "player_id"})
    if "nfl_id" in core.columns:
        core = core.rename(columns={"nfl_id": "player_id"})

    roster_cols = [
        c
        for c in [
            "game_id",
            "play_id",
            "player_id",
            "player_name",
            "player_position",
            "player_role",
            "player_side",
        ]
        if c in core.columns
    ]
    roster = core[roster_cols].drop_duplicates()

    join_keys = ["game_id", "play_id", "player_id"]
    join_keys = [k for k in join_keys if k in coords.columns and k in roster.columns]

    coords = coords.merge(
        roster,
        on=join_keys,
        how="left",
    )

    return coords

# Data loader for core (shared by Play Deep Dive) ----------------------------------

@st.cache_data(show_spinner=True)
def load_core_raw() -> pd.DataFrame:
    """Read merged_core.csv once and lowercase column names."""
    if not CORE_CSV.exists():
        raise FileNotFoundError(
            f"merged_core.csv not found at {CORE_CSV}. Run 01_merge_pipeline.py first."
        )
    df = pd.read_csv(CORE_CSV)
    df.columns = [c.lower() for c in df.columns]
    return df


# Play table for Deep Dive (1 row per gameId/playId) --------------------------------

@st.cache_data(show_spinner=True)
def load_play_table() -> pd.DataFrame:
    """
    Build a 1-row-per-play table with a human-readable label
    for the Play Deep Dive dropdown.

    Safe even if you only have game_id / play_id and nothing else.
    """
    df = load_core_raw()

    game_col = _pick_col(df, ["game_id_std", "game_id"])
    play_col = _pick_col(df, ["play_id_std", "play_id"])

    if not game_col or not play_col:
        return pd.DataFrame()

    # Try to pull a few metadata columns if they exist
    off_col = _pick_col(df, ["offense_team_std", "offense_team", "posteam"])
    def_col = _pick_col(df, ["defense_team_std", "defense_team", "defteam"])
    down_col = _pick_col(df, ["down_std", "down"])
    ytg_col = _pick_col(df, ["yards_to_go_std", "yards_to_go", "ydstogo"])
    desc_col = _pick_col(df, ["description_std", "play_description", "play_desc", "desc"])

    meta_cols = [c for c in [off_col, def_col, down_col, ytg_col, desc_col] if c]

    if meta_cols:
        # groupby first row for each play
        agg_dict = {c: "first" for c in meta_cols}
        plays = (
            df.groupby([game_col, play_col], as_index=False)
              .agg(agg_dict)
        )
    else:
        # no extra columns, just unique game/play pairs
        plays = df[[game_col, play_col]].drop_duplicates().copy()

    # Standardize column names
    rename_map = {game_col: "game_id_std", play_col: "play_id_std"}
    if off_col:
        rename_map[off_col] = "offense_team_std"
    if def_col:
        rename_map[def_col] = "defense_team_std"
    if down_col:
        rename_map[down_col] = "down_std"
    if ytg_col:
        rename_map[ytg_col] = "yards_to_go_std"
    if desc_col:
        rename_map[desc_col] = "description_std"

    plays = plays.rename(columns=rename_map)

    def fmt_down(d):
        try:
            d = int(d)
        except Exception:
            return ""
        return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(d, f"{d}th")

    labels = []
    for _, r in plays.iterrows():
        gid = r.get("game_id_std", np.nan)
        pid = r.get("play_id_std", np.nan)
        off = str(r.get("offense_team_std", "") or "").upper()
        deff = str(r.get("defense_team_std", "") or "").upper()
        down = r.get("down_std", np.nan)
        ytg = r.get("yards_to_go_std", np.nan)
        desc = str(r.get("description_std", "") or "")

        try:
            gid_int = int(gid)
            pid_int = int(pid)
            id_part = f"{gid_int}-{pid_int}"
        except Exception:
            id_part = "Play"

        short_desc = desc.strip()
        if len(short_desc) > 80:
            short_desc = short_desc[:77].rstrip() + "…"

        dd = ""
        if not pd.isna(down) and not pd.isna(ytg):
            dd = f"{fmt_down(down)} & {int(ytg)}"

        parts: list[str] = [id_part]
        if off or deff:
            parts.append(f"{off} vs {deff}")
        if dd:
            parts.append(dd)
        if short_desc:
            parts.append(short_desc)

        label = " · ".join(parts)
        labels.append(label)

    plays["play_label"] = labels
    plays = plays.sort_values(["game_id_std", "play_id_std"])

    return plays


# Movement & spatial view ---------------------------------------------------------

def view_movement_spatial():
    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader("Movement & spatial view")

    # --- 1) Look at what the user last selected on the assessments page ---
    last_cmp = st.session_state.get("last_comparison")
    last_single = st.session_state.get("last_single_player")

    default_wr_pid = None
    default_db_pid = None

    if last_cmp:
        default_wr_pid = last_cmp.get("pid_a")
        default_db_pid = last_cmp.get("pid_b")
    elif last_single:
        default_wr_pid = last_single.get("pid")

    # --- 2) Load coordinates + core info ---
    try:
        coords = load_coords()
    except Exception as e:
        st.error(str(e))
        return

    players = coords[coords["object_type"] == "player"].copy()
    ball = coords[coords["object_type"] == "ball"].copy()

    if players.empty:
        st.warning("No player trajectories found in coordinates.csv.")
        return

    # ---- side mapping ----
    if "player_side" in players.columns:
        side_col = "player_side"
    elif "side" in players.columns:
        side_col = "side"
    else:
        st.warning("player_side/side not found – cannot reliably separate WR/DB.")
        return

    players["ui_side"] = players[side_col].map(
        {"Offense": "WR", "Defense": "DB"}
    ).fillna(players[side_col])

    # ---- keep plays that contain both WR and DB ----
    valid_plays = []
    for pid, g in players.groupby("play_id"):
        sides = g["ui_side"].dropna().unique().tolist()
        if "WR" in sides and "DB" in sides:
            valid_plays.append(pid)

    if not valid_plays:
        st.warning("No plays found with both WR and DB.")
        return

    players = players[players["play_id"].isin(valid_plays)]
    ball = ball[ball["play_id"].isin(valid_plays)]

    # --- default play selection if WR/DB pair exists in same play ---
    default_play_id = None
    if default_wr_pid is not None and default_db_pid is not None:
        sub = players[players["player_id"].isin([default_wr_pid, default_db_pid])]
        counts = sub.groupby("play_id")["player_id"].nunique()
        matches = counts[counts == 2].index.tolist()
        if matches:
            default_play_id = int(matches[0])

    play_ids = sorted(valid_plays)
    default_play_index = play_ids.index(default_play_id) if default_play_id in play_ids else 0

    # ============ PLAY SELECTOR ============
    play_id = st.selectbox("Play ID", play_ids, index=default_play_index)

    # track data for selected play
    play_players = players[players["play_id"] == play_id].copy()
    play_ball = ball[ball["play_id"] == play_id].copy()

    # ========= WHO IS IN THIS PLAY? =========
    st.markdown("#### Players in this play")
    summary = (
        play_players[["player_id", "player_name", "player_position", "ui_side"]]
        .drop_duplicates()
        .sort_values(["ui_side", "player_position", "player_name"])
    )
    st.dataframe(summary, use_container_width=True)

    wr_pool = summary[summary["ui_side"] == "WR"]
    db_pool = summary[summary["ui_side"] == "DB"]

    if wr_pool.empty or db_pool.empty:
        st.warning("Selected play does not contain both WR and DB.")
        return

    # build dropdown labels
    def _label(r):
        return f"{r['player_name']} · {r['player_position']}"

    wr_options = {int(r.player_id): _label(r) for _, r in wr_pool.iterrows()}
    db_options = {int(r.player_id): _label(r) for _, r in db_pool.iterrows()}

    wr_keys = list(wr_options.keys())
    db_keys = list(db_options.keys())

    # defaults
    wr_default_index = wr_keys.index(default_wr_pid) if default_wr_pid in wr_keys else 0
    db_default_index = db_keys.index(default_db_pid) if default_db_pid in db_keys else 0

    c1, c2 = st.columns(2)
    with c1:
        wr_id = st.selectbox(
            "WR (Offense)",
            wr_keys,
            index=wr_default_index,
            format_func=lambda pid: wr_options[pid],
        )
    with c2:
        db_id = st.selectbox(
            "DB (Defense)",
            db_keys,
            index=db_default_index,
            format_func=lambda pid: db_options[pid],
        )

    wr_track = (
        play_players[play_players["player_id"] == wr_id]
        .sort_values("frame_id")
        .copy()
    )
    db_track = (
        play_players[play_players["player_id"] == db_id]
        .sort_values("frame_id")
        .copy()
    )

    if wr_track.empty or db_track.empty:
        st.warning("Missing tracking for selected WR/DB.")
        return

    # ============ SEPARATION PER FRAME ============
    sep = wr_track[["frame_id", "x", "y"]].merge(
        db_track[["frame_id", "x", "y"]],
        on="frame_id",
        suffixes=("_wr", "_db"),
    )
    sep["separation"] = np.sqrt(
        (sep["x_wr"] - sep["x_db"]) ** 2 + (sep["y_wr"] - sep["y_db"]) ** 2
    )

    wr_track = wr_track.merge(
        sep[["frame_id", "separation"]], on="frame_id", how="left"
    )

        # ============ ROUTES & FIELD VIEW ============
    # collect all x/y for auto-zoom
    all_x = list(wr_track["x"]) + list(db_track["x"])
    all_y = list(wr_track["y"]) + list(db_track["y"])
    if not play_ball.empty:
        all_x += list(play_ball["x"])
        all_y += list(play_ball["y"])

    xmin, xmax, ymin, ymax = _auto_bounds(all_x, all_y)

    fig = go.Figure()

    # --- base field rectangle ---
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=FIELD_LENGTH,
        y1=FIELD_WIDTH,
        line=dict(color="rgba(148,163,184,0.45)", width=1),
        fillcolor="rgba(15,118,110,0.70)",
        layer="below",
    )

    # soft yardlines every 10 yards
    field_shapes = []
    for yd in range(0, 121, 10):
        field_shapes.append(
            dict(
                type="line",
                x0=yd,
                x1=yd,
                y0=0,
                y1=FIELD_WIDTH,
                line=dict(color="rgba(148,163,184,0.35)", width=0.7),
                layer="below",
            )
        )
    fig.update_layout(shapes=tuple(fig.layout.shapes) + tuple(field_shapes))

    # --- WR route ---
    fig.add_trace(
        go.Scatter(
            x=wr_track["x"],
            y=wr_track["y"],
            mode="lines+markers",
            name=f"{wr_options[wr_id]} (WR)",
            customdata=np.stack(
                [wr_track["frame_id"], wr_track["separation"]], axis=-1
            ),
            hovertemplate="Frame %{customdata[0]}<br>Separation: %{customdata[1]:.2f} yd",
            line=dict(width=4, color="#38bdf8", shape="spline"),
            marker=dict(size=6, symbol="circle"),
        )
    )

    # --- DB route ---
    fig.add_trace(
        go.Scatter(
            x=db_track["x"],
            y=db_track["y"],
            mode="lines+markers",
            name=f"{db_options[db_id]} (DB)",
            customdata=np.stack([db_track["frame_id"]], axis=-1),
            hovertemplate="Frame %{customdata[0]}",
            line=dict(width=3, dash="dash", color="#f97316", shape="spline"),
            marker=dict(size=6, symbol="diamond"),
        )
    )

    # Ball path + target frames
    throw_frame: int | None = None
    target_frame: int | None = None

    if not play_ball.empty:
        play_ball = play_ball.sort_values("frame_id").copy()

        throw_frame = int(play_ball["frame_id"].iloc[0])
        target_frame = int(play_ball["frame_id"].iloc[-1])

        # full ball trajectory
        fig.add_trace(
            go.Scatter(
                x=play_ball["x"],
                y=play_ball["y"],
                mode="lines+markers",
                name="Ball trajectory",
                customdata=np.stack([play_ball["frame_id"]], axis=-1),
                hovertemplate="Frame %{customdata[0]}",
                line=dict(width=2, dash="dot", color="#fde68a", shape="spline"),
                marker=dict(size=5, symbol="circle-open"),
            )
        )

        # throw origin
        throw = play_ball.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[throw["x"]],
                y=[throw["y"]],
                mode="markers+text",
                name="Throw frame",
                text=["Throw"],
                textposition="bottom center",
                marker=dict(size=11, symbol="arrow-up", color="#fbbf24"),
            )
        )

        # arrival / catch point
        arrival = play_ball.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[arrival["x"]],
                y=[arrival["y"]],
                mode="markers+text",
                name="Arrival frame",
                text=["Ball arrival"],
                textposition="top center",
                marker=dict(size=13, symbol="star", color="#22c55e"),
            )
        )

    # --- key leverage event frames (must be AFTER we know throw/arrival) ---
    first_open_frame: int | None = None
    max_sep_frame: int | None = None

    if not sep.empty:
        open_low = 2.0
        open_high = 3.0

        window = sep.copy()
        if throw_frame is not None:
            window = window[window["frame_id"] >= throw_frame]
        if target_frame is not None:
            window = window[window["frame_id"] <= target_frame]

        if not window.empty:
            # first frame in NFL open window
            open_rows = window[
                (window["separation"] >= open_low)
                & (window["separation"] <= open_high)
            ]
            if not open_rows.empty:
                first_open_frame = int(open_rows["frame_id"].iloc[0])

            # frame of max separation between throw and arrival
            max_idx = window["separation"].idxmax()
            max_sep_frame = int(window.loc[max_idx, "frame_id"])

    # --- leverage snapshots on the field view ---
    if first_open_frame is not None:
        wr_open = wr_track[wr_track["frame_id"] == first_open_frame]
        db_open = db_track[db_track["frame_id"] == first_open_frame]

        if not wr_open.empty and not db_open.empty:
            wr_o = wr_open.iloc[0]
            db_o = db_open.iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=[wr_o["x"]],
                    y=[wr_o["y"]],
                    mode="markers+text",
                    name="WR first open (2–3 yds)",
                    text=["WR first open"],
                    textposition="bottom center",
                    marker=dict(
                        size=10,
                        symbol="diamond",
                        line=dict(width=1, color="white"),
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[db_o["x"]],
                    y=[db_o["y"]],
                    mode="markers",
                    name="DB at WR first-open frame",
                    marker=dict(size=9, symbol="x"),
                )
            )

    if max_sep_frame is not None:
        wr_max = wr_track[wr_track["frame_id"] == max_sep_frame]
        db_max = db_track[db_track["frame_id"] == max_sep_frame]

        if not wr_max.empty and not db_max.empty:
            wr_m = wr_max.iloc[0]
            db_m = db_max.iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=[wr_m["x"]],
                    y=[wr_m["y"]],
                    mode="markers+text",
                    name="WR max separation",
                    text=["Max sep"],
                    textposition="top center",
                    marker=dict(
                        size=11,
                        symbol="triangle-up",
                        line=dict(width=1, color="white"),
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[db_m["x"]],
                    y=[db_m["y"]],
                    mode="markers",
                    name="DB at max-sep frame",
                    marker=dict(size=10, symbol="x-open"),
                )
            )

    # field styling + auto-zoom window
    fig.update_layout(
        title=f"Routes & Separation — Play {play_id}",
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        xaxis=dict(
            title="Field position (yards)",
            range=[xmin, xmax],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Sideline to sideline (yards)",
            range=[ymin, ymax],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",   # keep proportions correct
            scaleratio=1,
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0.5,
            xanchor="center",
            title_text="",
            font=dict(size=11),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


    # ============ SEPARATION SUMMARY ============
    st.markdown("#### Separation summary")

    if sep.empty:
        st.write("No overlapping WR/DB frames to compute separation.")
    else:
        avg_sep = sep["separation"].mean()
        min_sep = sep["separation"].min()
        max_sep = sep["separation"].max()

        st.write(
            f"**Avg separation:** {avg_sep:.2f} yds · "
            f"**Min:** {min_sep:.2f} · "
            f"**Max:** {max_sep:.2f}"
        )

        FRAMES_PER_SECOND = 10.0
        open_low = 2.0
        open_high = 3.0

        window = sep.copy()
        if throw_frame is not None:
            window = window[window["frame_id"] >= throw_frame]
        if target_frame is not None:
            window = window[window["frame_id"] <= target_frame]

        if not window.empty:
            total_frames = len(window)
            open_frames = window[
                (window["separation"] >= open_low)
                & (window["separation"] <= open_high)
            ]
            wide_open_frames = window[window["separation"] > open_high]

            time_total = total_frames / FRAMES_PER_SECOND
            time_open = len(open_frames) / FRAMES_PER_SECOND
            time_wide_open = len(wide_open_frames) / FRAMES_PER_SECOND

            pct_open = (
                len(open_frames) / total_frames * 100.0 if total_frames else 0.0
            )

            st.write(
                f"**Between throw and arrival:** {time_total:.2f} s window · "
                f"{time_open:.2f} s in NFL open (2–3 yds, {pct_open:.0f}% of window) · "
                f"{time_wide_open:.2f} s >3 yds (wide open)"
            )

            # scout-style leverage note
            note_parts = []
            if first_open_frame is not None:
                note_parts.append(
                    "WR earns NFL-open separation at least once before arrival"
                )
            else:
                note_parts.append(
                    "WR never enters the 2–3 yd NFL-open band before arrival"
                )

            if time_wide_open > 0:
                note_parts.append(
                    f"spends {time_wide_open:.2f}s truly wide open (>3 yds)"
                )
            else:
                note_parts.append("never gets >3 yds clear")

            if max_sep_frame is not None:
                note_parts.append(
                    "max separation occurs between throw and arrival window"
                )

            st.write("**Leverage note:** " + " · ".join(note_parts) + ".")
        else:
            st.write(
                "Throw/arrival frames fall outside the overlap window for WR/DB – "
                "cannot compute time in NFL open range."
            )

        st.caption(
        "In the PER-10 model this separation curve feeds the **Separation (S)** pillar, "
        "while the timing of the throw, ball flight and WR/DB triggers feed "
        "**Anticipation (A)** and **Eyes (tracking)**."
        )


        # ============ SEPARATION OVER TIME ===================
    st.markdown("#### Separation over time")

    if sep.empty:
        st.info("Not enough overlapping frames to draw separation over time.")
    else:
        # Use the shared helper for consistency with other pages
        fig_sep = plot_separation_timeline(
            frames_wr=wr_track,
            frames_db=db_track,
            target_frame=target_frame,  # can be None, helper handles it
        )
        st.plotly_chart(fig_sep, use_container_width=True)

        st.caption(
            "This curve shows **when** the WR earns leverage, how long they stay "
            "within the NFL open window (2–3 yards), and the space available at the "
            "moment of the throw and at ball arrival."
        )


    # ============ SPEED PROFILE – WR vs DB ====================
    st.markdown("#### Speed profile – WR vs DB")

    def _add_speed(track):
        t = track.sort_values("frame_id").copy()
        t["dx"] = t["x"].diff()
        t["dy"] = t["y"].diff()
        t["dt"] = t["frame_id"].diff().replace(0, np.nan)
        t["speed_raw"] = np.sqrt(t["dx"] ** 2 + t["dy"] ** 2) / t["dt"]
        t["speed"] = t["speed_raw"].rolling(3, center=True).mean()
        return t

    wr_speed = _add_speed(wr_track)
    db_speed = _add_speed(db_track)

    if wr_speed["speed"].notna().any() or db_speed["speed"].notna().any():
        fig_speed = go.Figure()

        fig_speed.add_trace(
            go.Scatter(
                x=wr_speed["frame_id"],
                y=wr_speed["speed"],
                mode="lines",
                name="WR speed",
                line=dict(width=3),
                hovertemplate="Frame %{x}<br>Speed %{y:.2f}",
            )
        )

        fig_speed.add_trace(
            go.Scatter(
                x=db_speed["frame_id"],
                y=db_speed["speed"],
                mode="lines",
                name="DB speed",
                line=dict(width=3, dash="dash"),
                hovertemplate="Frame %{x}<br>Speed %{y:.2f}",
            )
        )

        fig_speed.update_layout(
            template="plotly_dark",
            plot_bgcolor="#020617",
            paper_bgcolor="#020617",
            xaxis_title="Frame",
            yaxis_title="Speed (yd/frame)",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_speed, use_container_width=True)

        st.caption(
            "The speed profile highlights **anticipation**, **pace control**, and "
            "**recovery bursts**. WR acceleration patterns often reveal early "
            "anticipation, while DB speed changes reflect leverage maintenance "
            "and recovery technique."
        )
    else:
        st.info("Could not compute valid speeds for this WR/DB pair.")
