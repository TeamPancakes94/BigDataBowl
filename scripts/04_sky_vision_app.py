# scripts/06_sky_vision_app.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# Paths
# ------------------
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
POST_DIR = OUT_DIR / "posterior"

PILLARS_CSV = POST_DIR / "posterior_pillars.csv"
OVERALL_CSV = POST_DIR / "posterior_overall.csv"
CORE_CSV = OUT_DIR / "merged_core.csv"

# canonical pillar order (for tables)
WR_PILLARS = ["anticipation", "execution", "separation", "innovation", "eyes"]
DB_PILLARS = ["coverage", "reaction"]
ALL_PILLARS = WR_PILLARS + [p for p in DB_PILLARS if p not in WR_PILLARS]



# Data loading
# ------------------
@st.cache_data(show_spinner=True)
def load_data():
    """Load posterior pillars, overall, and roster, and build a clean player pool."""
    if not PILLARS_CSV.exists() or not OVERALL_CSV.exists():
        raise FileNotFoundError(
            "Posterior files not found. Run 02_extract_pillars.py and 03_update_posterior.py first."
        )
    if not CORE_CSV.exists():
        raise FileNotFoundError(
            "merged_core.csv not found. Run 01_merge_pipeline.py first."
        )

    pillars = pd.read_csv(PILLARS_CSV)
    overall = pd.read_csv(OVERALL_CSV)
    core = pd.read_csv(CORE_CSV)

    # normalize column names
    for df in (pillars, overall, core):
        df.columns = [c.lower() for c in df.columns]

    # core has nfl_id; rename to player_id
    if "nfl_id" in core.columns:
        core = core.rename(columns={"nfl_id": "player_id"})

    # minimal roster (names + positions)
    roster = (
        core[["player_id", "player_name", "player_position"]]
        .dropna(subset=["player_id"])
        .drop_duplicates()
    )

    # attach roster info to posterior dfs
    pillars = pillars.merge(roster, on="player_id", how="left")
    overall = overall.merge(roster, on="player_id", how="left")

    # "pool" = all players that actually have an overall score (so WR/DB only)
    pool = (
        overall[["player_id", "player_name", "player_position", "side"]]
        .dropna(subset=["player_id", "side"])
        .drop_duplicates()
        .sort_values(["side", "player_position", "player_name"])
    )

    # create a label for dropdowns: "Name · POS · side"
    pool["label"] = pool.apply(
        lambda r: f"{r.player_name} · {r.player_position or ''} · {r.side}",
        axis=1,
    )

    return pillars, overall, pool



# Helper functions
# ---------------------------
def pillar_order_for_side(side: str) -> List[str]:
    return WR_PILLARS if side == "WR" else DB_PILLARS


def player_choices_for_side(pool: pd.DataFrame, side: str) -> pd.DataFrame:
    """Return subset of pool for a given side, with at least 1 row."""
    sub = pool[pool["side"] == side].copy()
    return sub.sort_values("player_name")


def select_player(label: str, pool_side: pd.DataFrame, key: str):
    """Generic selectbox that returns (player_id, player_name) or (None, None) if empty."""
    if pool_side.empty:
        st.warning(f"No players available for {label}. Check that your posterior files are built.")
        return None, None

    options = pool_side["label"].tolist()
    default_index = 0 if options else None
    selected = st.selectbox(label, options=options, index=default_index, key=key)
    row = pool_side.loc[pool_side["label"] == selected].iloc[0]
    return int(row["player_id"]), row["player_name"]


def scores_for_player(
    pillars: pd.DataFrame, player_id: int, side: str
) -> Dict[str, float]:
    """Return pillar -> score_1_10 dict for a given player + side."""
    sub = pillars[(pillars["player_id"] == player_id) & (pillars["side"] == side)]
    if sub.empty:
        return {}

    # mean 1–10 score per pillar (in case multiple plays)
    ser = (
        sub.groupby("pillar")["score_1_10"]
        .mean()
        .astype(float)
        .reindex(ALL_PILLARS)
    )
    return ser.dropna().to_dict()


def advanced_pillars_for_player(pillars: pd.DataFrame, player_id: int, side: str):
    """Return advanced stats (alpha, beta, mean, ci) table for a player+side."""
    sub = pillars[(pillars["player_id"] == player_id) & (pillars["side"] == side)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["pillar", "alpha", "beta", "mean", "score_1_10", "ci_low", "ci_high"])
    # already in posterior_pillars.csv
    cols = ["pillar", "alpha", "beta", "mean", "score_1_10", "ci_low", "ci_high"]
    existing = [c for c in cols if c in sub.columns]
    return sub[existing].sort_values("pillar")


def overall_row(overall: pd.DataFrame, player_id: int, side: str):
    sub = overall[(overall["player_id"] == player_id) & (overall["side"] == side)]
    return sub.iloc[0] if not sub.empty else None


def winner_label(name_a: str, score_a: float, name_b: str, score_b: float) -> str:
    """Simple winner logic: higher 1–10 wins; tie if close or both NaN."""
    if np.isnan(score_a) and np.isnan(score_b):
        return ""
    if np.isnan(score_a):
        return name_b
    if np.isnan(score_b):
        return name_a
    if abs(score_a - score_b) < 0.25:
        return "tie"
    return name_a if score_a > score_b else name_b



# main dashboard (assessment + comparison)
# -----------------------------------------------------
def view_single_player(pillars: pd.DataFrame, overall: pd.DataFrame, pool: pd.DataFrame):
    st.subheader("Player assessment")

    side_choice = st.radio("Side", ["WR", "DB"], horizontal=True)

    pool_side = player_choices_for_side(pool, side_choice)

    pid, name = select_player("Player", pool_side, key="single_player")
    if pid is None:
        return

    ov = overall_row(overall, pid, side_choice)

    col1, col2 = st.columns(2)
    with col1:
        val = ov["overall_0_100"] if (ov is not None and "overall_0_100" in ov and pd.notna(ov["overall_0_100"])) else None
        st.metric("Overall (0–100)", f"{val:.1f}" if val is not None else "—")
    with col2:
        val = ov["per10"] if (ov is not None and "per10" in ov and pd.notna(ov["per10"])) else None
        st.metric("PER-10", f"{val:.1f}" if val is not None else "—")

    # Basic pillar score table (1–10)
    scores = scores_for_player(pillars, pid, side_choice)
    if not scores:
        st.info("No pillar scores found for this player/side.")
    else:
        rows = [
            {"pillar": p, "score_1_10": scores.get(p, np.nan)}
            for p in pillar_order_for_side(side_choice)
        ]
        st.markdown("#### Pillar scores (1–10)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Advanced stats (beta, alpha, mean, CI)
    with st.expander("Advanced pillar stats (alpha, beta, mean, CI)"):
        adv = advanced_pillars_for_player(pillars, pid, side_choice)
        if adv.empty:
            st.write("No advanced stats available for this player.")
        else:
            st.dataframe(adv, use_container_width=True)


def view_comparison(pillars: pd.DataFrame, overall: pd.DataFrame, pool: pd.DataFrame):
    st.subheader("Player comparison")

    layout = st.radio(
        "Comparison layout",
        ["1v1 matchup (WR vs DB / WR vs WR / DB vs DB)", "Multi-player grid (same side, up to 8)"],
        index=0,
    )

    if layout.startswith("1v1"):
        # ---------------- 1v1 MATCHUP ----------------
        comp_type = st.selectbox(
            "Comparison type",
            ["WR vs DB", "WR vs WR", "DB vs DB"],
            index=0,
        )

        if comp_type == "WR vs DB":
            side_a, side_b = "WR", "DB"
        elif comp_type == "WR vs WR":
            side_a = side_b = "WR"
        else:
            side_a = side_b = "DB"

        pool_a = player_choices_for_side(pool, side_a)
        pool_b = player_choices_for_side(pool, side_b)

        col1, col2 = st.columns(2)
        with col1:
            pid_a, name_a = select_player("Player A", pool_a, key="cmp_a")
        with col2:
            pid_b, name_b = select_player("Player B", pool_b, key="cmp_b")

        if pid_a is None or pid_b is None:
            return
        if pid_a == pid_b and side_a == side_b:
            st.warning("Please choose two different players.")
            return

        ov_a = overall_row(overall, pid_a, side_a)
        ov_b = overall_row(overall, pid_b, side_b)

        mcols = st.columns(4)
        with mcols[0]:
            val = ov_a["overall_0_100"] if (ov_a is not None and "overall_0_100" in ov_a and pd.notna(ov_a["overall_0_100"])) else None
            st.metric(f"{name_a} overall", f"{val:.1f}" if val is not None else "—")
        with mcols[1]:
            val = ov_a["per10"] if (ov_a is not None and "per10" in ov_a and pd.notna(ov_a["per10"])) else None
            st.metric(f"{name_a} PER-10", f"{val:.1f}" if val is not None else "—")
        with mcols[2]:
            val = ov_b["overall_0_100"] if (ov_b is not None and "overall_0_100" in ov_b and pd.notna(ov_b["overall_0_100"])) else None
            st.metric(f"{name_b} overall", f"{val:.1f}" if val is not None else "—")
        with mcols[3]:
            val = ov_b["per10"] if (ov_b is not None and "per10" in ov_b and pd.notna(ov_b["per10"])) else None
            st.metric(f"{name_b} PER-10", f"{val:.1f}" if val is not None else "—")

        # pillar comparison table
        scores_a = scores_for_player(pillars, pid_a, side_a)
        scores_b = scores_for_player(pillars, pid_b, side_b)

        rows = []
        for pillar in ALL_PILLARS:
            s_a = scores_a.get(pillar, np.nan)
            s_b = scores_b.get(pillar, np.nan)
            if np.isnan(s_a) and np.isnan(s_b):
                continue
            winner = winner_label(name_a, s_a, name_b, s_b)
            rows.append(
                {
                    "pillar": pillar,
                    name_a: s_a,
                    name_b: s_b,
                    "winner": winner,
                }
            )

        st.markdown("#### Pillar comparison table (who won the rep type)")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.write("No pillar scores found for this matchup.")

    else:
        # ---------------- MULTI-PLAYER GRID ----------------
        side = st.radio("Side", ["WR", "DB"], horizontal=True)

        pool_side = player_choices_for_side(pool, side)
        if pool_side.empty:
            st.warning("No players available for this side.")
            return

        # multiselect up to 8 players
        options = pool_side["label"].tolist()
        default = options[:4]  # pick first few by default
        selected_labels = st.multiselect(
            "Players (up to 8)",
            options=options,
            default=default,
            max_selections=8,
        )

        if not selected_labels:
            st.info("Select at least one player to show the grid.")
            return

        sel = pool_side[pool_side["label"].isin(selected_labels)]
        pid_list = sel["player_id"].astype(int).tolist()
        name_list = sel["player_name"].tolist()

        # build pillar x player table
        pillars_order = pillar_order_for_side(side)
        data = {}
        for pid, name in zip(pid_list, name_list):
            scores = scores_for_player(pillars, pid, side)
            col = [scores.get(p, np.nan) for p in pillars_order]
            data[name] = col

        grid = pd.DataFrame(data, index=pillars_order).reset_index().rename(
            columns={"index": "pillar"}
        )

        st.markdown("#### Multi-player pillar grid (1–10)")
        st.dataframe(grid, use_container_width=True)


# other tabs
# --------------------------
def view_advanced_stats(pillars: pd.DataFrame, overall: pd.DataFrame):
    st.subheader("Advanced stats – full posterior tables")

    st.markdown("##### Posterior pillars (per player × side × pillar)")
    st.dataframe(pillars, use_container_width=True)

    st.markdown("##### Posterior overall (per player × side)")
    st.dataframe(overall, use_container_width=True)


def view_movement_spatial():
    st.subheader("Movement & spatial view")
    st.write(
        """
        This tab will show route trails and spacing for selected plays – 
        for example, WR and DB paths from ball snap to catch/arrival, 
        colored by separation and leverage. 
        
        (Hook here for your future matplotlib/plotly animations using tracking data.)
        """
    )


def view_play_example():
    st.subheader("Play example breakdown")
    st.write(
        """
        Here we will feature a high-profile rep (e.g., a critical 3rd-down throw),
        and show:
        - the Sky Vision pillar scores for that rep
        - why the WR or DB “won” the matchup
        - the tracking-based movement view, aligned with video.
        
        For now this is a placeholder with text only.
        """
    )


def view_pillar_guide():
    st.subheader("Pillar guide")

    st.markdown("**Anticipation (WR)** – How early and cleanly the receiver anticipates the coverage and adjusts their route.")
    st.markdown("**Execution (WR)** – Route technique: stem, break, and timing relative to the QB and coverage.")
    st.markdown("**Separation (WR)** – How much space the WR creates and maintains vs the defender.")
    st.markdown("**Innovation (WR)** – Tempo changes, second moves, and improvisation to win late in the route.")
    st.markdown("**Eyes (WR)** – Ball-tracking and ability to find/track the ball in the air (1–10).")
    st.markdown("---")
    st.markdown("**Coverage (DB)** – How tightly and consistently the defender stays connected to the route.")
    st.markdown("**Reaction (DB)** – How quickly the DB reacts to the WR’s breaks and the QB’s actions.")


# main page
# -------------------------------
def main():
    st.set_page_config(page_title="Sky Vision", layout="wide")

    # --- sidebar: table of contents ---
    with st.sidebar:
        st.header("Table of contents")
        page = st.radio(
            "Select a page:",
            ["Main dashboard", "Advanced stats", "Movement & spatial view", "Play example", "Pillar guide"],
            index=0,
        )

    st.title("Sky Vision")

    try:
        pillars, overall, pool = load_data()
    except Exception as e:
        st.error(str(e))
        return

    # route to the selected page
    if page == "Main dashboard":
        mode = st.radio("Mode", ["Player assessment", "Player comparison"], horizontal=True)
        if mode == "Player assessment":
            view_single_player(pillars, overall, pool)
        else:
            view_comparison(pillars, overall, pool)

    elif page == "Advanced stats":
        view_advanced_stats(pillars, overall)

    elif page == "Movement & spatial view":
        view_movement_spatial()

    elif page == "Play example":
        view_play_example()

    elif page == "Pillar guide":
        view_pillar_guide()


if __name__ == "__main__":
    main()
