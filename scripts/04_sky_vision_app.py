# sky_vision_app.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sky_views import view_movement_spatial  # Movement & Spatial tab
from sky_play_deep_dive import view_play_deep_dive  # Play Deep Dive tab
from sky_overview import view_welcome  # landing page with hero + pillars


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sky Vision",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- GLOBAL CSS: DARK NAVY + GLOW ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Teko:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Teko', sans-serif;
}
\/* HERO BOX THAT MATCHES PLAY DEEP DIVE STYLE */
.sky-hero-box {
    background: #05060d;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 2rem 2.4rem;
    border-radius: 22px;
    margin-top: 2rem;
    margin-bottom: 2.5rem;

    box-shadow:
        0 0 60px rgba(56,189,248,0.12),
        inset 0 0 40px rgba(56,189,248,0.06);
}
.sky-hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.sky-hero-body {
    font-size: 1rem;
    color: #d1d5db;
    margin-bottom: 1.4rem;
}
.sky-hero-chips {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.sky-chip {
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Hide sidebar */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

body {
    background: radial-gradient(circle at top, #111827 0, #020617 55%, #000 100%);
    color: #e5e7eb;
}

.block-container {
    max-width: 1200px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 0.5rem !important;
}

/* -------- TOP BAR / LOGO -------- */
.sky-top-bar {
    position: relative;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    padding: 1.2rem 0 0.9rem 0;
    margin-bottom: 0.4rem;
}

.sky-top-bar::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 1px;
    background: linear-gradient(
        90deg,
        rgba(15,23,42,0.0),
        rgba(148,163,184,0.32),
        rgba(56,189,248,0.75),
        rgba(148,163,184,0.32),
        rgba(15,23,42,0.0)
    );
}

.sky-logo {
    font-family: 'Teko', sans-serif;
    font-size: 2.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #f9fafb;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.sky-logo-tagline {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #cbd5f5;
    margin-top: -0.25rem;
}

/* -------- NAV TABS (CENTERED, ONLY ACTIVE GLOWS) -------- */
.sky-nav-row {
    width: 100%;
    margin-top: 1.8rem;
    margin-bottom: 1.4rem;
}

/* base style for all nav buttons */
.sky-nav-row div.stButton > button {
    border-radius: 999px !important;
    padding: 0.8rem 2.9rem !important;
    min-width: 210px !important;

    background: #020617 !important;
    border: 1px solid rgba(148,163,184,0.55) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    color: #e5e7eb !important;
    text-transform: uppercase;
    white-space: nowrap;

    box-shadow:
        0 10px 24px rgba(15,23,42,0.9),
        0 0 0 1px rgba(15,23,42,0.9);
    transition:
        transform 130ms ease,
        box-shadow 130ms ease,
        background 130ms ease,
        border-color 130ms ease,
        filter 130ms ease;
}

/* hover for all tabs */
.sky-nav-row div.stButton > button:hover {
    background: #0b1120 !important;
    transform: translateY(-1px);
    border-color: rgba(148,163,184,0.9) !important;
    box-shadow:
        0 14px 30px rgba(15,23,42,1.0),
        0 0 10px rgba(15,23,42,0.6);
    filter: none;
}

/* center the 4 columns inside the nav row */
.sky-nav-row [data-testid="stHorizontalBlock"] {
    justify-content: center;
    gap: 2.0rem;
}

.sky-nav-row [data-testid="column"] {
    flex: 0 0 auto;
    display: flex;
    justify-content: center;
}

/* -------- SUBPAGE MINI TITLE -------- */
.sky-subtitle {
    font-size: 1.3rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-top: 0.3rem;
    margin-bottom: 0.75rem;
}

/* ===== HERO CARD (SKY VISION OVERVIEW) ===== */

.sky-hero-wrapper {
    position: relative;
    margin-top: 1.5rem;
    margin-bottom: 2.0rem;
    padding: 2.4rem 2.2rem;
    border-radius: 26px;
    overflow: hidden;

    background:
        radial-gradient(circle at 0% 0%, rgba(56,189,248,0.30), transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(129,140,248,0.32), transparent 55%),
        linear-gradient(135deg, #020617, #020617 40%, #020617 60%, #020617);
    border: 1px solid rgba(148,163,184,0.65);
    box-shadow:
        0 26px 70px rgba(15,23,42,0.95),
        0 0 40px rgba(15,23,42,0.9);
}

.sky-hero-kicker {
    font-size: 0.78rem;
    letter-spacing: 0.19em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 0.9rem;
    opacity: 0.95;
}

.sky-hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    color: #f9fafb;
}

.sky-hero-body {
    font-size: 0.96rem;
    line-height: 1.7;
    max-width: 46rem;
    color: #e5e7eb;
    margin-bottom: 1.6rem;
}

.sky-hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
}

.sky-hero-chip {
    font-size: 0.78rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(129,140,248,0.85);
    background: rgba(15,23,42,0.85);
    backdrop-filter: blur(10px);
    white-space: nowrap;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #e5e7eb;
}

.sky-hero-wrapper::before {
    content: "";
    position: absolute;
    inset: -30%;
    background:
        radial-gradient(circle at 10% 0%, rgba(56,189,248,0.18), transparent 55%),
        radial-gradient(circle at 80% 100%, rgba(236,72,153,0.22), transparent 55%);
    opacity: 0.85;
    filter: blur(12px);
    z-index: -1;
}

/* ===== SECTION DIVIDERS ===== */

.sky-section-divider {
    border: 0;
    border-top: 1px solid rgba(148,163,184,0.35);
    margin: 1.8rem 0 1.4rem 0;
}

.sky-top-divider {
    margin-top: 2.0rem;
}

/* ===== PILLAR GRID BUTTONS ===== */

.pillar-grid {
    margin-top: 0.75rem;
    margin-bottom: 1.2rem;
}

.pillar-grid div.stButton > button {
    width: 100%;
    text-align: left;
    padding: 0.9rem 1.1rem;
    border-radius: 16px;

    background: rgba(15,23,42,0.95);
    border: 1px solid rgba(148,163,184,0.5);
    color: #e5e7eb;
    font-size: 0.82rem;
    line-height: 1.4;
    text-transform: none;
    letter-spacing: 0.02em;

    box-shadow:
        0 16px 36px rgba(15,23,42,0.9),
        0 0 0 1px rgba(15,23,42,1.0);
    transition:
        transform 140ms ease,
        box-shadow 140ms ease,
        border-color 140ms ease,
        background 140ms ease,
        filter 140ms ease;
    white-space: normal;
}

.pillar-grid div.stButton > button:hover {
    transform: translateY(-1px);
    background: #020617;
    border-color: rgba(56,189,248,0.85);
    box-shadow:
        0 18px 40px rgba(15,23,42,1.0),
        0 0 14px rgba(56,189,248,0.9);
    filter: drop-shadow(0 0 12px rgba(56,189,248,0.6));
}

.pillar-grid div.stButton > button:focus-visible {
    outline: none;
    border-color: rgba(56,189,248,1.0);
    box-shadow:
        0 0 0 1px rgba(56,189,248,0.9),
        0 0 0 4px rgba(56,189,248,0.45);
}
</style>
""",
    unsafe_allow_html=True,
)


def rubric_tier(score: float) -> str:
    """Map a 0–10 pillar score to a qualitative tier."""
    if pd.isna(score):
        return ""
    if score >= 9:
        return "Elite"
    if score >= 7:
        return "Strong"
    if score >= 4:
        return "Solid"
    if score >= 2:
        return "Developing"
    return "Early stage"


# ---------- SIMPLE PAGE ROUTER ----------
if "page" not in st.session_state:
    st.session_state.page = "home"  # "home", "assess", "movement", "advanced", "play"


def go(page: str):
    st.session_state.page = page
    st.rerun()


def render_top_bar():
    """Logo + glowy nav tabs."""
    st.markdown(
        """
<div class="sky-top-bar">
  <div>
    <div class="sky-logo">
      <span>SKY VISION</span>
      <span></span>
    </div>
    <div class="sky-logo-tagline">
      Redefining Player Evaluation
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Centered nav row
    st.markdown('<div class="sky-nav-row">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    page = st.session_state.page

    # Map page -> which tab index should glow (1-based)
    if page in ("home", "assess"):
        active_idx = 1
    elif page == "movement":
        active_idx = 2
    elif page == "advanced":
        active_idx = 3
    elif page == "play":
        active_idx = 4
    else:
        active_idx = 1

    with col1:
        if st.button("Player Assessments", key="nav_assess_main"):
            go("assess")

    with col2:
        if st.button("Movement & Spatial", key="nav_movement_main"):
            go("movement")

    with col3:
        if st.button("Advanced Metrics", key="nav_advanced_main"):
            go("advanced")

    with col4:
        if st.button("Play Deep Dive", key="nav_play_main"):
            go("play")

    st.markdown("</div>", unsafe_allow_html=True)

    # Inject CSS to glow ONLY the active tab (nth column)
    st.markdown(
        f"""
<style>
.sky-nav-row [data-testid="column"]:nth-of-type({active_idx}) div.stButton > button {{
    background: linear-gradient(90deg, #2563eb 0%, #38bdf8 50%, #1d4ed8 100%) !important;
    border-color: rgba(56,189,248,0.95) !important;
    color: #f9fafb !important;
    box-shadow:
        0 20px 48px rgba(37,99,235,0.85),
        0 0 24px rgba(56,189,248,0.95);
    transform: translateY(-2px);
    filter: drop-shadow(0 0 10px rgba(56,189,248,0.7));
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_subpage_header(title: str):
    back_col, title_col = st.columns([1, 6])
    with back_col:
        if st.button("← Main", key=f"back_{title}"):
            go("home")
    with title_col:
        st.markdown(
            f'<div class="sky-subtitle">{title}</div>', unsafe_allow_html=True
        )


# ---------- DATA & HELPERS ----------
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

TRAITS_CSV = OUT_DIR / "per10_traits.csv"
BAYES_CSV = OUT_DIR / "bayesian_player_ratings.csv"
CORE_CSV = OUT_DIR / "merged_core.csv"
PILLARS_CSV = OUT_DIR / "merged_pillars.csv"

WR_PILLARS = [
    "anticipation",
    "separation",
    "execution",
    "eyes",
    "innovation",
    "improv",  # optional 6th dimension
]

DB_PILLARS = [
    "anticipation",
    "separation",
    "execution",
    "eyes",
    "innovation",
    "improv",
]

ALL_PILLARS = WR_PILLARS  # unified framework; DB uses the same codes


@st.cache_data(show_spinner=True)
def load_data():
    if not TRAITS_CSV.exists():
        raise FileNotFoundError(
            f"per10_traits.csv not found at {TRAITS_CSV}. "
            "Run 02_extract_pillars.py first."
        )
    if not BAYES_CSV.exists():
        raise FileNotFoundError(
            f"bayesian_player_ratings.csv not found at {BAYES_CSV}. "
            "Run model_update.py after 02_extract_pillars.py."
        )
    if not CORE_CSV.exists():
        raise FileNotFoundError(
            f"merged_core.csv not found at {CORE_CSV}. "
            "Run 01_merge_pipeline.py first."
        )

    traits = pd.read_csv(TRAITS_CSV)
    ratings = pd.read_csv(BAYES_CSV)
    core = pd.read_csv(CORE_CSV)
    pillars = pd.read_csv(PILLARS_CSV)

    for df in (traits, ratings, core, pillars):
        df.columns = [c.lower() for c in df.columns]

    # Normalise IDs
    if "nfl_id" in traits.columns:
        traits = traits.rename(columns={"nfl_id": "player_id"})
    if "nfl_id" in core.columns:
        core = core.rename(columns={"nfl_id": "player_id"})
    if "nfl_id" in ratings.columns and "player_id" not in ratings.columns:
        ratings = ratings.rename(columns={"nfl_id": "player_id"})

    roster = (
        core[["player_id", "player_name", "player_position"]]
        .dropna(subset=["player_id"])
        .drop_duplicates()
    )

    traits = traits.merge(roster, on="player_id", how="left")
    ratings = ratings.merge(roster, on="player_id", how="left")

    # Side mapping (WR / DB)
    if "player_side" in traits.columns:
        traits["side"] = traits["player_side"].map(
            {"Offense": "WR", "Defense": "DB"}
        ).fillna(traits["player_side"])
    else:
        traits["side"] = np.nan

    side_map = (
        traits[["player_id", "side"]]
        .dropna(subset=["side"])
        .drop_duplicates()
    )
    ratings = ratings.merge(side_map, on="player_id", how="left")

    # Per-10 360 from traits (rep-level) → per player-side
    if "per10_360" in traits.columns:
        per10_players = (
            traits.groupby(["player_id", "side"])["per10_360"]
            .mean()
            .reset_index()
            .rename(columns={"per10_360": "per10"})
        )
        ratings = ratings.merge(per10_players, on=["player_id", "side"], how="left")
    else:
        ratings["per10"] = np.nan

    # pillar rows for player assessments

    PILLAR_COLS = {
        "anticipation": "a",
        "separation": "s",
        "execution": "e",
        "eyes": "eyes",
        "innovation": "innovation",
        "improv": "improv",
    }

    rows = []
    for pillar_name, col in PILLAR_COLS.items():
        if col not in traits.columns:
            continue  # skip if somehow missing
        tmp = traits[["player_id", "side", col]].copy()
        tmp = tmp.rename(columns={col: "score_1_10"})
        tmp["pillar"] = pillar_name
        rows.append(tmp)

    if rows:
        pillars = pd.concat(rows, ignore_index=True)
    else:
        pillars = pd.DataFrame(
            columns=["player_id", "side", "pillar", "score_1_10"]
        )

    pool = (
        ratings[
            ["player_id", "player_name", "player_position", "side", "per10", "overall_0_100"]
        ]
        .dropna(subset=["player_id", "side"])
        .drop_duplicates()
    )

    pool["label"] = pool.apply(
        lambda r: f"{r.player_name} · {r.player_position or ''} · {r.side or ''}",
        axis=1,
    )

    overall = ratings  # naming for compatibility with views

    return pillars, overall, pool


def pillar_order_for_side(side: str) -> List[str]:
    # unified for now; if you ever want DB-specific ordering, change here
    return WR_PILLARS


def player_choices_for_side(pool: pd.DataFrame, side: str) -> pd.DataFrame:
    sub = pool[pool["side"] == side].copy()
    return sub.sort_values("player_name")


def select_player(label: str, pool_side: pd.DataFrame, key: str):
    if pool_side.empty:
        st.warning(
            f"No players available for {label}. Check that your posterior files are built."
        )
        return None, None

    options = pool_side["label"].tolist()
    default_index = 0 if options else None
    selected = st.selectbox(label, options=options, index=default_index, key=key)
    row = pool_side.loc[pool_side["label"] == selected].iloc[0]
    return int(row["player_id"]), row["player_name"]


def scores_for_player(
    pillars: pd.DataFrame, player_id: int, side: str
) -> Dict[str, float]:
    sub = pillars[(pillars["player_id"] == player_id) & (pillars["side"] == side)]
    if sub.empty:
        return {}

    ser = (
        sub.groupby("pillar")["score_1_10"]
        .mean()
        .astype(float)
        .reindex(ALL_PILLARS)
    )
    return ser.dropna().to_dict()


def overall_row(overall: pd.DataFrame, player_id: int, side: str):
    sub = overall[(overall["player_id"] == player_id) & (overall["side"] == side)]
    return sub.iloc[0] if not sub.empty else None


def winner_label(name_a: str, score_a: float, name_b: float, score_b: float) -> str:
    if np.isnan(score_a) and np.isnan(score_b):
        return ""
    if np.isnan(score_a):
        return name_b
    if np.isnan(score_b):
        return name_a
    if abs(score_a - score_b) < 0.25:
        return "tie"
    return name_a if score_a > score_b else name_b


# ---------- ADVANCED METRICS + LEAGUE-WIDE INSIGHTS ----------
def view_advanced_stats(pillars: pd.DataFrame, overall: pd.DataFrame):
    st.subheader("Insights & advanced metrics")

    st.markdown(
        "View **league-wide PER-10 patterns** and dig into the "
        "**posterior tables and archetype map** behind PER-10 and PER-10 360."
    )

    # ============= LEAGUE-WIDE INSIGHTS =============
    if ("pillar" in pillars.columns) and ("score_1_10" in pillars.columns):

        st.markdown("### League-wide insights (pillar vs PER-10 / overall)")

        st.markdown(
            "- **Eyes (tracking)** is the strongest single predictor of PER-10 360.\n"
            "- **Execution (finish)** shows moderate predictive value.\n"
            "- **Separation** is less correlated than expected, suggesting that *how* separation "
            "is created matters more than *how much*."
        )

        # Decide which rating we treat as primary target
        if "per10" in overall.columns:
            y_col = "per10"
            y_label = "PER-10 360"
        elif "overall_0_100" in overall.columns:
            y_col = "overall_0_100"
            y_label = "Overall (0–100)"
        else:
            y_col = None
            y_label = "rating"

        # Average pillar score per player × side × pillar
        by_player_pillar = (
            pillars.dropna(subset=["score_1_10"])
            .groupby(["player_id", "side", "pillar"], as_index=False)["score_1_10"]
            .mean()
        )

        # Attach rating
        if y_col is not None and y_col in overall.columns:
            overall_small = (
                overall[["player_id", "side", y_col]]
                .dropna(subset=["player_id", "side", y_col])
                .drop_duplicates()
            )

            merged = by_player_pillar.merge(
                overall_small,
                on=["player_id", "side"],
                how="left",
            ).dropna(subset=[y_col])

            # --- 1) Correlation: which pillar tracks the rating best? ---
            corr_rows = []
            for pillar_name, g in merged.groupby("pillar"):
                # Need at least 2 distinct scores for a sensible correlation
                if len(g) < 3 or g["score_1_10"].nunique() <= 1:
                    continue
                try:
                    r = float(
                        np.corrcoef(g["score_1_10"].to_numpy(),
                                    g[y_col].to_numpy())[0, 1]
                    )
                except Exception:
                    continue
                if np.isnan(r):
                    continue
                corr_rows.append({"pillar": pillar_name, "corr": r})

            if corr_rows:
                corr_df = pd.DataFrame(corr_rows)
                corr_df["abs_corr"] = corr_df["corr"].abs()
                corr_df = corr_df.sort_values("abs_corr", ascending=False)

                pretty_labels = {
                    "anticipation": "Anticipation (pre-snap)",
                    "separation": "Separation (space)",
                    "execution": "Execution (finish)",
                    "eyes": "Eyes (tracking)",
                    "innovation": "Innovation (design)",
                    "improv": "Improv (in-play)",
                }

                top = corr_df.iloc[0]
                top_name = pretty_labels.get(top["pillar"], top["pillar"].title())

                st.markdown(
                    f"- **Most predictive pillar right now:** "
                    f"**{top_name}** vs **{y_label}** "
                    f"(correlation ≈ `{top['corr']:.2f}`)."
                )
                st.markdown(
                    "- The table below shows how strongly each pillar tracks the final "
                    f"{y_label.lower()} rating across all player-sides."
                )

                show_corr = corr_df[["pillar", "corr"]].copy()
                show_corr["pillar"] = show_corr["pillar"].map(
                    lambda p: pretty_labels.get(p, p.title())
                )
                show_corr["corr"] = show_corr["corr"].round(2)
                show_corr = show_corr.rename(
                    columns={"pillar": "Pillar", "corr": f"Corr vs {y_label}"}
                )
                st.dataframe(show_corr, use_container_width=True)

                # --- Add interpretability notes for sparse / low-variance pillars ---
                st.markdown(
                    """
<div style="margin-top: 0.6rem; font-size: 0.85rem; color: #cbd5f5;">
<strong>Note:</strong> Some pillars such as <strong>Anticipation</strong> and 
<strong>Innovation</strong> do not appear in the correlation table. 
This is intentional:
<ul style='margin-top:0.25rem;'>
<li><strong>Anticipation</strong> scores cluster in a narrow band across players 
(low statistical variance), so league-wide correlations do not meaningfully distinguish them.</li>
<li><strong>Innovation</strong> events are tagged only when a player creates a new in-rep 
solution. They are sparse by design, highlighting ceiling plays rather than broad trends.</li>
</ul>
These pillars remain highly diagnostic in <em>player profiles</em> and 
<em>1v1 matchup analysis</em> even if they do not drive league-wide correlation.
</div>
                    """,
                    unsafe_allow_html=True,
                )

            # --- 2) Rating distribution by side (WR vs DB) ---
            if "side" in overall.columns:
                dist = overall.dropna(subset=[y_col, "side"])
                if not dist.empty:
                    st.markdown("#### Rating distribution by side")

                    fig_dist = px.histogram(
                        dist,
                        x=y_col,
                        color="side",
                        nbins=25,
                        barmode="overlay",
                        opacity=0.65,
                        marginal="box",
                        labels={y_col: y_label, "side": "Side"},
                    )

                    fig_dist.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#020617",
                        paper_bgcolor="#020617",
                        margin=dict(l=40, r=40, t=40, b=40),
                        font=dict(family="Teko, sans-serif", size=14),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0,
                            title_text="",
                        ),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            # --- 3) Average pillar scores by side (WR vs DB) ---
            if "side" in pillars.columns:
                st.markdown("#### Average pillar scores by side")

                avg = (
                    pillars.dropna(subset=["score_1_10"])
                    .groupby(["side", "pillar"], as_index=False)["score_1_10"]
                    .mean()
                )

                pivot = avg.pivot(index="pillar", columns="side", values="score_1_10")
                pivot = pivot.rename(
                    index={
                        "anticipation": "Anticipation (A)",
                        "separation": "Separation (S)",
                        "execution": "Execution (E)",
                        "eyes": "Eyes",
                        "innovation": "Innovation",
                        "improv": "Improv (I)",
                    }
                )
                st.dataframe(pivot.round(2), use_container_width=True)

        st.markdown('<hr class="sky-section-divider">', unsafe_allow_html=True)

    # ============= EXISTING ADVANCED METRICS =============
    st.markdown("### Posterior tables & PER-10 archetype map")

    # --- Raw tables for judges ---
    with st.expander(
        "Posterior pillars table (per player × side × pillar)", expanded=False
    ):
        st.dataframe(pillars, use_container_width=True, height=350)

    with st.expander("Posterior overall table (per player × side)", expanded=False):
        st.dataframe(overall, use_container_width=True, height=350)

    st.markdown("#### PER-10 archetype map – pillar vs PER-10 360")

    # ---- Basic guards ----
    if "pillar" not in pillars.columns:
        st.info("No 'pillar' column found in pillars table – cannot build archetype map.")
        return

    score_col = "score_1_10" if "score_1_10" in pillars.columns else None
    if score_col is None:
        st.info("Could not find a numeric score column (expected 'score_1_10').")
        return

    # ---- Pillar selector ----
    unique_pillars = sorted(pillars["pillar"].dropna().unique().tolist())
    if not unique_pillars:
        st.info("No pillars found in pillars table – cannot build archetype map.")
        return

    # Default to Eyes if present, else first pillar
    default_idx = 0
    for i, p in enumerate(unique_pillars):
        if "eye" in str(p).lower():
            default_idx = i
            break

    selected_pillar = st.selectbox(
        "Choose pillar for X-axis",
        unique_pillars,
        index=default_idx,
    )

    pillar_df = pillars[pillars["pillar"] == selected_pillar].copy()
    if pillar_df.empty:
        st.info(
            f"No rows found for pillar '{selected_pillar}' – cannot build archetype map."
        )
        return

    pillar_summary = (
        pillar_df.groupby(["player_id", "side"])[score_col]
        .mean()
        .reset_index()
        .rename(columns={score_col: "pillar_score"})
    )

    # y-axis = PER-10 360 if available, else overall_0_100
    y_col = "per10" if "per10" in overall.columns else None
    if y_col is None and "overall_0_100" in overall.columns:
        y_col = "overall_0_100"

    if y_col is None:
        st.info("Could not find PER-10 360 or overall_0_100 columns for the Y-axis.")
        return

    overall_cols = [
        c
        for c in [
            "player_id",
            "side",
            "player_name",
            "player_position",
            "per10",
            "overall_0_100",
        ]
        if c in overall.columns
    ]
    overall_unique = overall[overall_cols].drop_duplicates()

    summary = pillar_summary.merge(
        overall_unique,
        on=["player_id", "side"],
        how="left",
    )
    summary = summary.dropna(subset=["pillar_score", y_col])
    if summary.empty:
        st.info(
            f"Not enough data to build the {selected_pillar} vs PER-10 360 archetype map."
        )
        return

    summary["label"] = summary.apply(
        lambda r: f"{r.get('player_name', 'Player')} · "
                  f"{r.get('player_position', '')} · "
                  f"{r.get('side', '')}",
        axis=1,
    )

    # Colors: cyan WR, violet DB
    color_map = {"WR": "#0ea5e9", "DB": "#a855f7"}

    # Axis labels based on selection
    x_label = f"{selected_pillar.capitalize()} pillar (1–10)"
    y_label = "PER-10 360" if y_col == "per10" else "Overall (0–100)"

    fig = px.scatter(
        summary,
        x="pillar_score",
        y=y_col,
        color="side" if "side" in summary.columns else None,
        color_discrete_map=color_map,
        hover_name="label",
        hover_data={
            "pillar_score": ":.2f",
            y_col: ":.2f",
        },
        labels={
            "pillar_score": x_label,
            "per10": "PER-10 360",
            "overall_0_100": "Overall (0–100)",
            "side": "Side",
        },
    )

    fig.update_traces(
        marker=dict(
            size=9,
            opacity=0.9,
            line=dict(width=0.6, color="#020617"),
        )
    )

    # Medians to create archetype quadrants
    x_med = float(summary["pillar_score"].median())
    y_med = float(summary[y_col].median())

    fig.add_vline(
        x=x_med,
        line_width=1,
        line_dash="dot",
        line_color="rgba(148,163,184,0.7)",
    )
    fig.add_hline(
        y=y_med,
        line_width=1,
        line_dash="dot",
        line_color="rgba(148,163,184,0.7)",
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        margin=dict(l=40, r=40, t=60, b=70),
        font=dict(family="Teko, sans-serif", size=14),
        xaxis=dict(
            title=x_label,
            title_standoff=30,
            showgrid=True,
            gridcolor="rgba(31,41,55,0.5)",
            zeroline=False,
        ),
        yaxis=dict(
            title=y_label,
            showgrid=True,
            gridcolor="rgba(31,41,55,0.5)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            title_text="",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Each point is a player-side (WR or DB). X-axis = **{selected_pillar} pillar (1–10)**. "
        f"Y-axis = **{y_label}**. Top-right = players who are strong on this pillar "
        "and high in overall Ball IQ. Top-left = players who win more with overall IQ "
        "than this specific pillar; bottom-right = players who lean more on this pillar "
        "than their overall IQ would suggest."
    )


# ---------- PLAYER ASSESSMENT VIEWS ----------
def view_single_player(pillars: pd.DataFrame, overall: pd.DataFrame, pool: pd.DataFrame):
    st.subheader("Player assessment")

    side_choice = st.radio("Side", ["WR", "DB"], horizontal=True)

    pool_side = player_choices_for_side(pool, side_choice)

    pid, name = select_player("Player", pool_side, key="single_player")
    if pid is None:
        return

    st.session_state["last_single_player"] = {
        "pid": int(pid),
        "name": name,
        "side": side_choice,
    }

    ov = overall_row(overall, pid, side_choice)

    col1, col2 = st.columns(2)
    with col1:
        val = (
            ov["overall_0_100"]
            if (ov is not None and "overall_0_100" in ov and pd.notna(ov["overall_0_100"]))
            else None
        )
        st.metric("Overall (0–100)", f"{val:.1f}" if val is not None else "—")
    with col2:
        val = ov["per10"] if (ov is not None and "per10" in ov and pd.notna(ov["per10"])) else None
        st.metric("PER-10 360", f"{val:.1f}" if val is not None else "—")

    # -------- pillar scores for this player --------
    scores = scores_for_player(pillars, pid, side_choice)
    if not scores:
        st.info("No pillar scores found for this player/side.")
        return

    pretty_labels = {
        "anticipation": "Anticipation (pre-snap)",
        "separation": "Separation (space)",
        "execution": "Execution (finish)",
        "eyes": "Eyes (tracking)",
        "innovation": "Innovation (design)",
        "improv": "Improv (in-play)",
    }

    order = pillar_order_for_side(side_choice)

    rows = []
    missing_codes = []

    for p in order:
        if p in scores:
            rows.append(
                {
                    "pillar_code": p,
                    "pillar": pretty_labels.get(p, p.title()),
                    "score_1_10": float(scores[p]),
                }
            )
        else:
            missing_codes.append(p)

    if not rows:
        st.info("No non-NaN pillar scores for this player.")
        return

    df_pillars = pd.DataFrame(rows)

    # -------- population percentiles & rubric tiers --------
    pop = pillars[pillars["side"] == side_choice].copy()

    percentiles: list[float] = []
    for _, r in df_pillars.iterrows():
        code = r["pillar_code"]
        val = r["score_1_10"]
        dist = pop.loc[pop["pillar"] == code, "score_1_10"].dropna()
        if dist.empty:
            percentiles.append(np.nan)
        else:
            pct = float((dist <= val).mean() * 100.0)
            percentiles.append(pct)

    df_pillars["percentile"] = percentiles
    df_pillars["tier"] = df_pillars["score_1_10"].apply(rubric_tier)

    # -------- within-player z-scores (relative strengths/weaknesses) --------
    vals = df_pillars["score_1_10"].astype(float)
    if vals.nunique() >= 2:
        mu = float(vals.mean())
        sigma = float(vals.std(ddof=0))
        if sigma > 0:
            df_pillars["z_within_player"] = (vals - mu) / sigma
        else:
            df_pillars["z_within_player"] = 0.0
    else:
        df_pillars["z_within_player"] = 0.0

    # -------- narrative: calling card & growth lane --------
    calling_text = ""
    growth_text = ""

    if len(df_pillars) >= 2 and df_pillars["z_within_player"].notna().any():
        # Calling card = highest z-score pillar
        cc_row = df_pillars.loc[df_pillars["z_within_player"].idxmax()]
        # Growth lane = lowest z-score pillar
        gl_row = df_pillars.loc[df_pillars["z_within_player"].idxmin()]

        cc_pct = (
            f"~{cc_row['percentile']:.0f}th percentile"
            if pd.notna(cc_row["percentile"])
            else "—"
        )
        gl_pct = (
            f"~{gl_row['percentile']:.0f}th percentile"
            if pd.notna(gl_row["percentile"])
            else "—"
        )

        calling_text = (
            f"**Calling card:** {cc_row['pillar']} "
            f"({cc_pct} for {side_choice}s, {cc_row['tier']})."
        )
        growth_text = (
            f"**Growth lane:** {gl_row['pillar']} "
            f"({gl_pct} for {side_choice}s, {gl_row['tier']})."
        )

    if calling_text or growth_text:
        st.markdown(f"{calling_text} {growth_text}")

    # -------- bar chart (sorted low→high for visual) --------
    df_plot = df_pillars.sort_values("score_1_10", ascending=True)

    fig = px.bar(
        df_plot,
        x="score_1_10",
        y="pillar",
        orientation="h",
        text="score_1_10",
        range_x=[0, 10],
        labels={
            "score_1_10": "Pillar score (1–10)",
            "pillar": "",
        },
    )

    palette = ["#38bdf8", "#22c55e", "#eab308", "#f97316", "#a855f7", "#ec4899"]
    fig.update_traces(
        marker=dict(color=palette[: len(df_plot)]),
        texttemplate="%{text:.1f}",
        textposition="outside",
        insidetextanchor="middle",
    )

    fig.update_layout(
        title=f"PER-10 Ball IQ profile — {name}",
        template="plotly_dark",
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Teko, sans-serif", size=15),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(31,41,55,0.6)",
            dtick=1,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed",
        ),
    )

    st.markdown("#### PER-10 Ball IQ profile (pillars 1–10)")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note: Some pillars (Separation, Innovation, Improv) only appear when tagged on applicable reps. "
        "Their sparsity reflects the underlying labels, not missing model components. Only pillars that currently have graded PER-10 data are shown in the chart."
    )

    if missing_codes:
        _ = [pretty_labels.get(p, p.title()) for p in missing_codes]

    # -------- population table --------
    with st.expander("Pillar scores vs population", expanded=False):
        show = df_pillars[["pillar", "score_1_10", "percentile", "tier"]].copy()
        show["score_1_10"] = show["score_1_10"].round(1)
        show["percentile"] = show["percentile"].round(1)
        st.dataframe(show, use_container_width=True)


def view_comparison(pillars: pd.DataFrame, overall: pd.DataFrame, pool: pd.DataFrame):
    st.subheader("Player comparison")

    layout = st.radio(
        "Comparison layout",
        ["1v1 matchup (WR vs DB / WR vs WR / DB vs DB)", "Multi-player grid (same side, up to 8)"],
        index=0,
    )

    # ---------- 1v1 MATCHUP ----------
    if layout.startswith("1v1"):
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

        st.session_state["last_comparison"] = {
            "pid_a": int(pid_a),
            "name_a": name_a,
            "side_a": side_a,
            "pid_b": int(pid_b),
            "name_b": name_b,
            "side_b": side_b,
        }

        mcols = st.columns(4)
        with mcols[0]:
            val = (
                ov_a["overall_0_100"]
                if (ov_a is not None and "overall_0_100" in ov_a and pd.notna(ov_a["overall_0_100"]))
                else None
            )
            st.metric(f"{name_a} overall", f"{val:.1f}" if val is not None else "—")
        with mcols[1]:
            val = ov_a["per10"] if (ov_a is not None and "per10" in ov_a and pd.notna(ov_a["per10"])) else None
            st.metric(f"{name_a} PER-10 360", f"{val:.2f}" if val is not None else "—")
        with mcols[2]:
            val = (
                ov_b["overall_0_100"]
                if (ov_b is not None and "overall_0_100" in ov_b and pd.notna(ov_b["overall_0_100"]))
                else None
            )
            st.metric(f"{name_b} overall", f"{val:.1f}" if val is not None else "—")
        with mcols[3]:
            val = ov_b["per10"] if (ov_b is not None and "per10" in ov_b and pd.notna(ov_b["per10"])) else None
            st.metric(f"{name_b} PER-10 360", f"{val:.2f}" if val is not None else "—")

        scores_a = scores_for_player(pillars, pid_a, side_a)
        scores_b = scores_for_player(pillars, pid_b, side_b)

        pretty_labels = {
            "anticipation": "Anticipation (pre-snap)",
            "separation": "Separation (space)",
            "execution": "Execution (finish)",
            "eyes": "Eyes (tracking)",
            "innovation": "Innovation (design)",
            "improv": "Improv (in-play)",
        }

        rows = []
        pillars_for_chart: list[str] = []
        a_vals: list[float] = []
        b_vals: list[float] = []

        for pillar_code in ALL_PILLARS:
            s_a = scores_a.get(pillar_code, np.nan)
            s_b = scores_b.get(pillar_code, np.nan)
            if np.isnan(s_a) and np.isnan(s_b):
                continue

            winner = winner_label(name_a, s_a, name_b, s_b)
            rows.append(
                {
                    "pillar": pillar_code,  # keep code; we'll map to label later
                    name_a: s_a,
                    name_b: s_b,
                    "winner": winner,
                }
            )

            pillars_for_chart.append(pillar_code)
            a_vals.append(s_a if not np.isnan(s_a) else np.nan)
            b_vals.append(s_b if not np.isnan(s_b) else np.nan)

        # ---------- NEW: matchup hinge + percentiles/tiers ----------
        hinge_desc = ""
        enhanced_rows = []

        pop = pillars.copy()

        for r in rows:
            code = r["pillar"]
            s_a = r[name_a]
            s_b = r[name_b]

            def pct_for(side, val):
                dist = pop[(pop["side"] == side) & (pop["pillar"] == code)]["score_1_10"].dropna()
                if dist.empty or pd.isna(val):
                    return np.nan
                return float((dist <= val).mean() * 100.0)

            pct_a = pct_for(side_a, s_a)
            pct_b = pct_for(side_b, s_b)

            r[f"pct_{name_a}"] = pct_a
            r[f"tier_{name_a}"] = rubric_tier(s_a)
            r[f"pct_{name_b}"] = pct_b
            r[f"tier_{name_b}"] = rubric_tier(s_b)

            enhanced_rows.append(r)

        rows = enhanced_rows

        # compute matchup hinge (largest absolute score gap where both valid)
        best_diff = -np.inf
        hinge_pillar = None
        hinge_edge_name = None

        for r in rows:
            s_a = r[name_a]
            s_b = r[name_b]
            if np.isnan(s_a) or np.isnan(s_b):
                continue
            diff = abs(s_a - s_b)
            if diff > best_diff:
                best_diff = diff
                hinge_pillar = r["pillar"]
                hinge_edge_name = name_a if s_a > s_b else name_b

        if hinge_pillar is not None and best_diff > 0:
            hinge_label = pretty_labels.get(hinge_pillar, hinge_pillar.title())
            hinge_desc = (
                f"Matchup hinge: **{hinge_label}** — "
                f"{hinge_edge_name} +{best_diff:.1f} on the 1–10 scale."
            )

        if hinge_desc:
            st.markdown(hinge_desc)

        st.markdown("#### PER-10 Ball IQ fingerprint")

        if pillars_for_chart:
            # Long-form dataframe for grouped bar chart
            chart_df = pd.DataFrame(
                {
                    "pillar": pillars_for_chart * 2,
                    "score": a_vals + b_vals,
                    "player": [name_a] * len(pillars_for_chart)
                    + [name_b] * len(pillars_for_chart),
                }
            )
            # Drop pillars where a player has no score
            chart_df = chart_df.dropna(subset=["score"])

            chart_df["pillar_label"] = chart_df["pillar"].map(pretty_labels)

            # Sort pillars by average score (so biggest pillars/mismatches float to top)
            order_df = (
                chart_df.groupby("pillar_label")["score"]
                .mean()
                .sort_values(ascending=True)
                .reset_index()
            )
            category_order = order_df["pillar_label"].tolist()

            fig = px.bar(
                chart_df,
                x="score",
                y="pillar_label",
                color="player",
                orientation="h",
                barmode="group",
                range_x=[0, 10],
                labels={
                    "score": "Pillar score (1–10)",
                    "pillar_label": "",
                    "player": "",
                },
            )

            fig.update_traces(
                text=chart_df["score"].round(1),
                textposition="outside",
            )

            fig.update_layout(
                plot_bgcolor="#020617",
                paper_bgcolor="#020617",
                font=dict(family="Teko, sans-serif", size=15),
                margin=dict(l=40, r=40, t=60, b=40),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(31,41,55,0.6)",
                    dtick=1,
                    zeroline=False,
                ),
                yaxis=dict(
                    showgrid=False,
                    categoryorder="array",
                    categoryarray=category_order,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Innovation and Improv events are tagged only on specific reps "
                "(creative or broken plays), so some pillars may be blank for a given player. "
                "They still feed the PER-10 model and matchup snapshot."
            )

            # ---------- upgraded comparison table ----------
            if rows:
                st.markdown("#### Pillar comparison table (PER-10 pillars)")

                comp_df = pd.DataFrame(rows)
                comp_df["pillar_label"] = comp_df["pillar"].map(pretty_labels)

                # round numeric cols
                for col in [name_a, name_b, f"pct_{name_a}", f"pct_{name_b}"]:
                    if col in comp_df.columns:
                        comp_df[col] = comp_df[col].round(2)

                cols_to_show = [
                    "pillar_label",
                    name_a,
                    f"pct_{name_a}",
                    f"tier_{name_a}",
                    name_b,
                    f"pct_{name_b}",
                    f"tier_{name_b}",
                    "winner",
                ]
                display = comp_df[cols_to_show].rename(
                    columns={
                        "pillar_label": "pillar",
                        name_a: f"{name_a} score",
                        f"pct_{name_a}": f"{name_a} pct",
                        f"tier_{name_a}": f"{name_a} tier",
                        name_b: f"{name_b} score",
                        f"pct_{name_b}": f"{name_b} pct",
                        f"tier_{name_b}": f"{name_b} tier",
                    }
                )

                st.dataframe(display, use_container_width=True)
        else:
            st.info("No pillar data available for this matchup.")

    # ---------- MULTI-PLAYER GRID ----------
    else:
        side = st.radio("Side", ["WR", "DB"], horizontal=True)

        pool_side = player_choices_for_side(pool, side)
        if pool_side.empty:
            st.warning("No players available for this side.")
            return

        options = pool_side["label"].tolist()
        default = options[:4]
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

        pillars_order = pillar_order_for_side(side)
        data = {}
        for pid, name in zip(pid_list, name_list):
            scores = scores_for_player(pillars, pid, side)
            col = [scores.get(p, np.nan) for p in pillars_order]
            data[name] = col

        grid = (
            pd.DataFrame(data, index=pillars_order)
            .reset_index()
            .rename(columns={"index": "pillar"})
        )

        st.markdown("#### Multi-player PER-10 pillar heatmap (1–10)")

        # ---- Heatmap: pillars (rows) × players (columns) ----
        heat = grid.set_index("pillar")

        fig = px.imshow(
            heat,
            text_auto=".1f",
            aspect="auto",
            zmin=0,
            zmax=10,
            color_continuous_scale=[
                "#020617",  # deep navy
                "#1d4ed8",  # blue
                "#22c55e",  # green
                "#eab308",  # gold
            ],
        )

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#020617",
            paper_bgcolor="#020617",
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(family="Teko, sans-serif", size=14),
            xaxis=dict(title="Player", side="top"),
            yaxis=dict(title="", autorange="reversed"),
            coloraxis_colorbar=dict(
                title="Score",
                ticks="outside",
                tickvals=[0, 2, 4, 6, 8, 10],
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Blank cells indicate pillars that have not been tagged often enough for that player "
            "yet (for example Innovation and Improv). These remain part of the model but are "
            "intentionally sparse ceiling traits."
        )

        with st.expander("Raw values (table view)", expanded=False):
            st.dataframe(grid, use_container_width=True)


# ---------- MAIN ----------
def main():
    render_top_bar()

    try:
        pillars, overall, pool = load_data()
    except Exception as e:
        st.error(str(e))
        return

    page = st.session_state.page

    if page == "home":
        view_welcome(pillars, overall)

    elif page == "assess":
        render_subpage_header("Player Assessments")
        mode = st.radio(
            "Mode",
            ["Player assessment", "Player comparison"],
            horizontal=True,
        )
        if mode == "Player assessment":
            view_single_player(pillars, overall, pool)
        else:
            view_comparison(pillars, overall, pool)

    elif page == "movement":
        render_subpage_header("Movement & Spatial View")
        view_movement_spatial()

    elif page == "advanced":
        render_subpage_header("Advanced Metrics")
        view_advanced_stats(pillars, overall)

    elif page == "play":
        render_subpage_header("Play Deep Dive")
        view_play_deep_dive()


if __name__ == "__main__":
    main()
