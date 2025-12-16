# scripts/sky_play_deep_dive.py
# Real-play PER-10 Deep Dive, using merged_core.csv + per10_traits.csv

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
CORE_CSV = OUT_DIR / "merged_core.csv"
TRAITS_CSV = OUT_DIR / "per10_traits.csv"   # <-- new


# ------------------ Helpers ------------------

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -------- Ball IQ context helpers (matchup snapshot) --------

PILLAR_COLS_DEEP_DIVE = {
    "anticipation": "a",
    "separation": "s",
    "execution": "e",
    "eyes": "eyes",
    "innovation": "innovation",
    "improv": "improv",
}

PRETTY_PILLAR_LABELS = {
    "anticipation": "Anticipation (pre-snap)",
    "separation": "Separation (space)",
    "execution": "Execution (finish)",
    "eyes": "Eyes (tracking)",
    "innovation": "Innovation (design)",
    "improv": "Improv (in-play)",
}


def _rubric_tier(score: float) -> str:
    """Tier labels roughly matching the rubric in the deck."""
    if pd.isna(score):
        return "—"
    if score >= 8.5:
        return "Elite"
    if score >= 7.0:
        return "Plus starter"
    if score >= 5.5:
        return "Solid starter"
    if score >= 4.0:
        return "Developing"
    return "Early stage"


@st.cache_data(show_spinner=False)
def load_pillar_population_for_deep_dive() -> pd.DataFrame:
    """
    Lightweight population table: player_id × side × pillar × score_1_10
    used for percentiles and matchup stories.
    """
    if not TRAITS_CSV.exists():
        return pd.DataFrame()

    traits = pd.read_csv(TRAITS_CSV)
    traits.columns = [c.lower() for c in traits.columns]

    if "nfl_id" in traits.columns:
        traits = traits.rename(columns={"nfl_id": "player_id"})

    # side mapping for WR/DB
    if "player_side" in traits.columns:
        traits["side"] = traits["player_side"].map(
            {"Offense": "WR", "Defense": "DB"}
        ).fillna(traits["player_side"])
    elif "side" not in traits.columns:
        traits["side"] = np.nan

    rows = []
    for pillar_name, col in PILLAR_COLS_DEEP_DIVE.items():
        if col not in traits.columns:
            continue
        tmp = traits[["player_id", "side", col]].copy()
        tmp = tmp.rename(columns={col: "score_1_10"})
        tmp["pillar"] = pillar_name
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["player_id", "side", "pillar", "score_1_10"])

    pop = pd.concat(rows, ignore_index=True)
    return pop


def _profile_for_player(pop: pd.DataFrame, pid: int, side: str) -> pd.DataFrame | None:
    """Return pillar score + percentile per pillar for a single player/side."""
    sub = pop[(pop["player_id"] == pid) & (pop["side"] == side)]
    if sub.empty:
        return None

    base = (
        sub.groupby("pillar")["score_1_10"]
        .mean()
        .reset_index()
        .rename(columns={"score_1_10": "score"})
    )

    records = []
    for _, r in base.iterrows():
        code = r["pillar"]
        val = r["score"]
        dist = pop[(pop["side"] == side) & (pop["pillar"] == code)]["score_1_10"].dropna()
        if dist.empty or pd.isna(val):
            pct = np.nan
        else:
            pct = float((dist <= val).mean() * 100.0)
        records.append({"pillar": code, "score": val, "percentile": pct})

    return pd.DataFrame(records)


def _calling_and_growth(profile: pd.DataFrame) -> tuple[dict | None, dict | None]:
    """
    Calling card = highest-percentile pillar.
    Growth lane = lowest-percentile pillar, only if <= 40th and different.
    """
    df = profile.dropna(subset=["percentile"]).copy()
    if df.empty:
        return None, None

    call_row = df.sort_values("percentile", ascending=False).iloc[0].to_dict()
    growth_row = df.sort_values("percentile", ascending=True).iloc[0].to_dict()

    if growth_row["pillar"] == call_row["pillar"] or growth_row["percentile"] > 40:
        growth_row = None

    return call_row, growth_row


def render_matchup_scout_summary(
    focus_wr_row: pd.Series | None,
    focus_db_row: pd.Series | None,
):
    """
    Render a short scout-style Ball IQ summary for the currently selected WR / DB.

    focus_wr_row / focus_db_row are rows from merged_core for this play, with at
    least: player_id, player_name, and player_side or side.
    """
    if focus_wr_row is None or focus_db_row is None:
        return

    pop = load_pillar_population_for_deep_dive()
    if pop.empty:
        return

    def _side_from_row(r):
        raw = r.get("side", r.get("player_side", ""))
        raw = str(raw)
        if raw == "Offense":
            return "WR"
        if raw == "Defense":
            return "DB"
        return raw or "WR"

    # WR
    wr_id = int(focus_wr_row["player_id"])
    wr_name = str(focus_wr_row.get("player_name", "WR"))
    wr_side = _side_from_row(focus_wr_row)

    # DB
    db_id = int(focus_db_row["player_id"])
    db_name = str(focus_db_row.get("player_name", "DB"))
    db_side = _side_from_row(focus_db_row)

    wr_profile = _profile_for_player(pop, wr_id, wr_side)
    db_profile = _profile_for_player(pop, db_id, db_side)
    if wr_profile is None or db_profile is None:
        return

    wr_call, wr_growth = _calling_and_growth(wr_profile)
    db_call, db_growth = _calling_and_growth(db_profile)

    # matchup hinge: biggest absolute score gap on a pillar both have
    merged = wr_profile.merge(db_profile, on="pillar", suffixes=("_wr", "_db"))
    merged = merged.dropna(subset=["score_wr", "score_db"])
    hinge_line = ""
    if not merged.empty:
        merged["diff"] = merged["score_wr"] - merged["score_db"]
        hinge = merged.iloc[merged["diff"].abs().argmax()]
        pillar_code = hinge["pillar"]
        diff = float(hinge["diff"])
        edge_name = wr_name if diff > 0 else db_name
        pretty = PRETTY_PILLAR_LABELS.get(pillar_code, pillar_code.title())
        hinge_line = (
            f"Biggest tilt: **{pretty}** — {edge_name} +{abs(diff):.1f} on the 1–10 scale."
        )

    def line_for_player(name, call, growth, side_label):
        parts = []
        if call is not None:
            pcode = call["pillar"]
            pretty = PRETTY_PILLAR_LABELS.get(pcode, pcode.title())
            tier = _rubric_tier(call["score"])
            pct = call["percentile"]
            parts.append(
                f"{name} calling card: **{pretty}** "
                f"({tier}, ~{pct:.0f}th percentile for {side_label}s)."
            )
        if growth is not None:
            pcode = growth["pillar"]
            pretty = PRETTY_PILLAR_LABELS.get(pcode, pcode.title())
            tier = _rubric_tier(growth["score"])
            pct = growth["percentile"]
            parts.append(
                f"Growth lane: **{pretty}** "
                f"({tier}, ~{pct:.0f}th percentile)."
            )
        return " ".join(parts)

    wr_line = line_for_player(wr_name, wr_call, wr_growth, wr_side)
    db_line = line_for_player(db_name, db_call, db_growth, db_side)

    st.markdown("### Matchup snapshot – Ball IQ context")
    st.write(wr_line)
    st.write(db_line)
    if hinge_line:
        st.write(hinge_line)


# ---------------- Core data loaders (merged_core only) ----------------

@st.cache_data(show_spinner=True)
def load_core_raw() -> pd.DataFrame:
    """Read merged_core.csv once and lowercase all column names."""
    if not CORE_CSV.exists():
        raise FileNotFoundError(
            f"merged_core.csv not found at {CORE_CSV}. "
            "Run 01_merge_pipeline.py before using Play Deep Dive."
        )
    df = pd.read_csv(CORE_CSV)
    df.columns = [c.lower() for c in df.columns]

    if "nfl_id" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"nfl_id": "player_id"})

    return df


@st.cache_data(show_spinner=True)
def load_play_table() -> pd.DataFrame:
    """
    Build a 1-row-per-play table with a readable label for the dropdown.

    Robust to merged_core.csv having only game_id/play_id (no extra metadata).
    """
    df = load_core_raw()

    game_col = _pick_col(df, ["game_id_std", "game_id"])
    play_col = _pick_col(df, ["play_id_std", "play_id"])

    # off_col = _pick_col(df, ["offense_team_std", "offense_team", "posteam"])
    # def_col = _pick_col(df, ["defense_team_std", "defense_team", "defteam"])

    down_col = _pick_col(df, ["down_std", "down"])
    ytg_col = _pick_col(df, ["yards_to_go_std", "yards_to_go", "ydstogo"])
    desc_col = _pick_col(df, ["description_std", "play_description", "play_desc", "desc"])

    if not game_col or not play_col:
        # Nothing we can do if we don't even have game/play ids
        return pd.DataFrame()

    # Build aggregation dict only for columns that actually exist
    agg_dict: dict[str, str] = {}
    for c in [down_col, ytg_col, desc_col]:
        if c:
            agg_dict[c] = "first"

    if agg_dict:
        # We have some metadata columns – groupby + agg
        plays = df.groupby([game_col, play_col], as_index=False).agg(agg_dict)
    else:
        # Only game/play available – just dedupe those two columns
        plays = df.drop_duplicates(subset=[game_col, play_col])[[game_col, play_col]].copy()

    # Standard names used downstream
    rename_map = {game_col: "game_id_std", play_col: "play_id_std"}
    #if off_col:
        #rename_map[off_col] = "offense_team_std"
    #if def_col:
        #rename_map[def_col] = "defense_team_std"
    if down_col:
        rename_map[down_col] = "down_std"
    if ytg_col:
        rename_map[ytg_col] = "yards_to_go_std"
    if desc_col:
        rename_map[desc_col] = "description_std"

    # Only rename columns that actually exist in plays
    rename_map = {k: v for k, v in rename_map.items() if k in plays.columns}
    plays = plays.rename(columns=rename_map)

    def fmt_down(d):
        try:
            d = int(d)
        except Exception:
            return ""
        return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(d, f"{d}th")

    labels = []
    for _, r in plays.iterrows():
        gid = int(r["game_id_std"])
        pid = int(r["play_id_std"])
        down = r.get("down_std", np.nan)
        ytg = r.get("yards_to_go_std", np.nan)
        desc = str(r.get("description_std", "") or "")

        # Kill literal "nan" text
        if desc.strip().lower() == "nan":
            desc = ""

        dd = ""
        if not pd.isna(down) and not pd.isna(ytg):
            dd = f"{fmt_down(down)} & {int(ytg)}"

        parts = [f"{gid}-{pid}"]
        if dd:
            parts.append(dd)
        if desc:
            parts.append(desc)

        label = " · ".join(parts)
        labels.append(label)

    plays["play_label"] = labels
    plays = plays.sort_values(["game_id_std", "play_id_std"])
    return plays


# PLAY DEEP DIVE VIEW --------------------------------

def play_deep_dive():
    """
    Play Deep Dive:
    Interactive WR vs DB play breakdown, using real NFL plays from merged_core.csv.
    """

    # ---------- HIGH PILLAR EXAMPLES (CARD) ----------
    st.markdown("### High pillar example plays")
    st.caption("Examples of how Sky Vision’s pillars appear in real NFL reps:")

    st.markdown(
        """
<div class="play-example-grid">

  <div class="play-panel play-example-card">
    <div class="play-example-pill">Eyes</div>
    <div class="play-example-title">Elite ball tracking</div>
    <div class="play-example-body">
      Early head turn and continuous tracking into a back-shoulder window.
    </div>
    <div class="play-example-result">
      Result: WR widens late, catches through contact, and finishes in bounds.
    </div>
  </div>

  <div class="play-panel play-example-card">
    <div class="play-example-pill">Anticipation</div>
    <div class="play-example-title">Leverage timing before the break</div>
    <div class="play-example-body">
      DB jumps leverage <strong>before</strong> the WR declares the break and beats him to the spot.
    </div>
    <div class="play-example-result">
      Result: Defender arrives first at the catch point.
    </div>
  </div>

  <div class="play-panel play-example-card">
    <div class="play-example-pill">Execution</div>
    <div class="play-example-title">Body control at the catch point</div>
    <div class="play-example-body">
      WR re-stacks after contact, stays square, and brings hands late through the DB.
    </div>
    <div class="play-example-result">
      Result: Wins contested catch with clean mechanics.
    </div>
  </div>

  <div class="play-panel play-example-card">
    <div class="play-example-pill">Innovation</div>
    <div class="play-example-title">Creative route adaptation</div>
    <div class="play-example-body">
      WR tempo-shifts, changes stride pattern, and generates a new throwing window.
    </div>
    <div class="play-example-result">
      Result: QB hits an unexpected window created by WR processing.
    </div>
  </div>

  <div class="play-panel play-example-card">
    <div class="play-example-pill">Improv Index</div>
    <div class="play-example-title">When the play breaks</div>
    <div class="play-example-body">
      QB scrambles, WR mirrors and bends into an open lane, DB recovers late.
    </div>
    <div class="play-example-result">
      Result: Positive play entirely created outside structure.
    </div>
  </div>

</div>
        """,
        unsafe_allow_html=True,
    )


    # ---------- CSS (KEEPING YOUR THEME) ----------
    st.markdown(
        """
<style>
.play-hero {
    border-radius: 1.5rem;
    padding: 1.6rem 1.9rem 1.8rem 1.9rem;
    background: #020617;
    border: 1px solid rgba(148,163,184,0.45);
    box-shadow:
        0 18px 40px rgba(15,23,42,0.9),
        0 0 0 1px rgba(15,23,42,0.9);
    margin-top: 1.3rem;
    margin-bottom: 1.7rem;
}
.play-hero-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.24em;
    color: #94a3b8;
    margin-bottom: 0.45rem;
}
.play-hero-title {
    font-size: 1.4rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #e5e7eb;
    margin-bottom: 0.7rem;
}
.play-hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.9rem;
}
.play-meta-pill {
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.6);
    font-size: 0.74rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #e5e7eb;
    background: radial-gradient(circle at top,
                rgba(30,64,175,0.42),
                #020617 55%,
                #000 100%);
}
.play-hero-body {
    font-size: 0.94rem;
    line-height: 1.55rem;
    color: #cbd5f5;
    max-width: 60rem;
}
.play-panel {
    border-radius: 1.25rem;
    padding: 1.3rem 1.5rem 1.35rem 1.5rem;
    background: #020617;
    border: 1px solid rgba(148,163,184,0.4);
    box-shadow:
        0 14px 32px rgba(15,23,42,0.85),
        0 0 0 1px rgba(15,23,42,0.9);
    margin-bottom: 1.3rem;
}
.play-panel-title {
    font-size: 0.92rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #e5e7eb;
    margin-bottom: 0.5rem;
}
.play-panel-sub {
    font-size: 0.86rem;
    color: #cbd5f5;
    line-height: 1.45rem;
    margin-bottom: 0.5rem;
}
.play-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0.45rem 0 0.65rem 0;
}
.play-pill-tag {
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.6);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #e5e7eb;
}
.play-pill-tag .code {
    font-weight: 600;
    margin-right: 0.25rem;
}
.play-timeline-step {
    margin-top: 0.6rem;
}
.play-timeline-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #e5e7eb;
    margin-bottom: 0.15rem;
}
.play-timeline-body {
    font-size: 0.83rem;
    color: #cbd5f5;
    line-height: 1.45rem;
}
.play-footnote {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: 0.65rem;
}
/* tiny spacing tweak for the intro card paragraphs */
.play-panel p {
    margin-bottom: 0.7rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
    """
<style>
/* GRID WRAPPER */
.play-example-grid {
    display: flex;
    flex-direction: column;
    gap: 0.65rem;   /* tighter spacing between cards */
    margin-top: 0.65rem;
}

/* two-column layout on wider screens */
@media (min-width: 900px) {
  .play-example-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.75rem;   /* slightly tighter grid gap */
  }
  .play-example-grid > .play-example-card:last-child {
      grid-column: 1 / -1;  /* center the last card across both columns */
  }
}

/* CARD STYLE (inherits play-panel shadow + rounding) */
.play-example-card {
    padding: 0.85rem 1.05rem 0.9rem 1.05rem;  /* tighter padding */
}

/* SMALL TOP PILL */
.play-example-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.10rem 0.55rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.6);
    font-size: 0.63rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #e5e7eb;
    margin-bottom: 0.35rem;
}

/* TITLE */
.play-example-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 0.15rem;   /* tighter */
}

/* BODY TEXT */
.play-example-body {
    font-size: 0.82rem;
    color: #cbd5f5;
    line-height: 1.3rem;   /* tighter line height */
    margin-bottom: 0.15rem;
}

/* RESULT TEXT */
.play-example-result {
    font-size: 0.78rem;
    font-style: italic;
    color: #9ca3af;
    margin-bottom: 0.0rem;   /* nearly no bottom gap */
}
</style>
    """,
    unsafe_allow_html=True,
)


    plays = load_play_table()
    if plays.empty:
        st.info(
            "No play-level data available yet – make sure merged_core.csv is built "
            "with game_id and play_id columns."
        )
        return

    st.markdown("### Select a real play from the model sample")
    st.caption("Play (gameId – playId · down & distance · description)")

    labels = plays["play_label"].tolist()

    # Default index = 0 (first play)
    default_index = 0

    # If a card was clicked on the Player Assessment tab, honor that
    target_gid = st.session_state.get("deep_dive_target_gid")
    target_pid = st.session_state.get("deep_dive_target_pid")

    if target_gid is not None and target_pid is not None:
        # Make sure we match the exact "gameId-playId" prefix
        target_prefix = f"{int(target_gid)}-{int(target_pid)}"
        for i, lbl in enumerate(labels):
            if lbl.startswith(target_prefix):
                default_index = i
                break

    selected_label = st.selectbox(
        "",
        options=labels,
        index=default_index,
        key="deep_dive_play_label",
    )



    # Once we've used it, clear the target so manual changes behave normally
    if "deep_dive_target_gid" in st.session_state:
        del st.session_state["deep_dive_target_gid"]
    if "deep_dive_target_pid" in st.session_state:
        del st.session_state["deep_dive_target_pid"]


    row = plays.loc[plays["play_label"] == selected_label].iloc[0]

    gid = int(row["game_id_std"])
    pid = int(row["play_id_std"])
    down = row.get("down_std", np.nan)
    ytg = row.get("yards_to_go_std", np.nan)

    core_full = load_core_raw()
    game_col = _pick_col(core_full, ["game_id_std", "game_id"])
    play_col = _pick_col(core_full, ["play_id_std", "play_id"])

    # extra context
    qtr_col = _pick_col(core_full, ["quarter_std", "quarter", "qtr"])
    yardline_col = _pick_col(core_full, ["yardline_std", "yardline_100", "yardline"])
    route_col = _pick_col(core_full, ["route_std", "route"])
    cov_col = _pick_col(core_full, ["coverage_std", "coverage"])
    wr_col = _pick_col(core_full, ["wr_name_std", "wr_name", "targeted_receiver"])
    db_col = _pick_col(core_full, ["db_name_std", "db_name", "primary_defender"])
    pass_col = _pick_col(core_full, ["pass_result_std", "passresult", "pass_result"])
    yards_col = _pick_col(core_full, ["yards_gained_std", "yards_gained"])

    # subset for this play
    if not game_col or not play_col:
        sub = pd.DataFrame()
    else:
        sub = core_full[(core_full[game_col] == gid) & (core_full[play_col] == pid)]

    def _first(col):
        if col and col in sub.columns and not sub[col].dropna().empty:
            return sub[col].dropna().iloc[0]
        return np.nan

    qtr = _first(qtr_col)
    yardline = _first(yardline_col)
    route = str(_first(route_col) or "")
    coverage = str(_first(cov_col) or "")
    if route.strip().lower() == "nan":
        route = ""
    if coverage.strip().lower() == "nan":
        coverage = ""

    wr_name = str(_first(wr_col) or "")
    db_name = str(_first(db_col) or "")
    pass_result = str(_first(pass_col) or "").upper()
    yards_gained = _first(yards_col)

    def fmt_down(d):
        try:
            d = int(d)
        except Exception:
            return ""
        return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(d, f"{d}th")

    dd = ""
    if not pd.isna(down) and not pd.isna(ytg):
        dd = f"{fmt_down(down)} & {int(ytg)}"

    qtr_label = f"Q{int(qtr)}" if not pd.isna(qtr) else ""
    yd_label = f"Yardline {int(yardline)}" if not pd.isna(yardline) else ""
    route_label = route or "boundary vertical concept"
    cov_label = coverage or "man / match coverage"

    if wr_name and db_name:
        matchup_label = f"{wr_name} (WR) vs {db_name} (DB)"
    elif wr_name:
        matchup_label = f"{wr_name} (targeted WR)"
    else:
        matchup_label = "Targeted WR vs primary DB"

    # meta pills
    pills = [f"Game {gid} · Play {pid}"]
    if qtr_label:
        pills.append(qtr_label)
    if yd_label:
        pills.append(yd_label)
    pills.append(route_label)
    pills.append(cov_label)
    pills_html = "".join(f'<div class="play-meta-pill">{p}</div>' for p in pills)

    # ---------- HERO ----------

    if dd:
        title_line = f"Game {gid} · Play {pid} · {dd}"
    else:
        title_line = f"Game {gid} · Play {pid}"

    st.markdown(
        f"""
<div class="play-hero">
  <div class="play-hero-label">
    PLAY DEEP DIVE · REAL REP FROM KAGGLE SAMPLE
  </div>

  <div class="play-hero-title">
    {title_line}
  </div>

  <div class="play-hero-meta">
    {pills_html}
  </div>

  <p class="play-hero-body">
    This is a real NFL rep drawn from the tracking sample used by Sky Vision. In the model this rep feeds the
    Ball IQ pillars: anticipation off the snap, separation down the stem,
    execution at the catch point, and how both players adjust once the ball is
    in the air.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Some sample plays in the tracking data do not include fully labeled WR/DB matchups. "
        "Sky Vision still logs the full pillar event timeline for these reps using tracking."
    )

    # ---------- PLAYERS INVOLVED ----------
    if not sub.empty:
        player_cols = [
            c
            for c in [
                "player_id",
                "player_name",
                "player_position",
                "player_side",
                "team",
                "side",
            ]
            if c in sub.columns
        ]
        if player_cols:
            players_tbl = sub[player_cols].drop_duplicates()

            sort_cols = [
                c
                for c in ["player_side", "side", "player_position", "player_name"]
                if c in players_tbl.columns
            ]
            if sort_cols:
                players_tbl = players_tbl.sort_values(sort_cols)

            st.markdown("#### Players involved in this play")
            st.dataframe(players_tbl, use_container_width=True)

    # ---------- FOCUS WR / DB (from merged_core) ----------
    side_col = "player_side" if "player_side" in sub.columns else None

    focus_wr_name = wr_name
    focus_db_name = db_name
    focus_wr_row = None
    focus_db_row = None

    last_single = st.session_state.get("last_single_player")
    default_wr_name_from_single = None
    default_db_name_from_single = None
    if last_single is not None:
        if last_single.get("side") == "WR":
            # We'll try to match this WR name in the dropdown below
            default_wr_name_from_single = last_single.get("name")
        elif last_single.get("side") == "DB":
            default_db_name_from_single = last_single.get("name")

    if side_col:
        off_players = sub[sub[side_col] == "Offense"]
        def_players = sub[sub[side_col] == "Defense"]

        pos_col = "player_position" if "player_position" in sub.columns else None

        if pos_col and not off_players.empty:
            wr_candidates = off_players[
                off_players[pos_col].isin(["WR", "TE", "RB", "FB"])
            ]
            if wr_candidates.empty:
                wr_candidates = off_players
        else:
            wr_candidates = off_players

        db_candidates = def_players

        col_wr, col_db = st.columns(2)
        with col_wr:
            if not wr_candidates.empty:
                wr_option_names = (
                    wr_candidates.get("player_name", pd.Series(dtype=str))
                    .dropna()
                    .unique()
                    .tolist()
                )

                # Default = 0, but try to match:
                # 1) player from Player Assessments, else
                # 2) the "wr_name" inferred earlier
                default_wr_idx = 0
                if default_wr_name_from_single and default_wr_name_from_single in wr_option_names:
                    default_wr_idx = wr_option_names.index(default_wr_name_from_single)
                elif wr_name and wr_name in wr_option_names:
                    default_wr_idx = wr_option_names.index(wr_name)

                wr_name_sel = st.selectbox(
                    "Focus receiver in this play",
                    options=wr_option_names,
                    index=default_wr_idx,
                )

                focus_wr_name = wr_name_sel or focus_wr_name

                if wr_name_sel:
                    focus_wr_row = wr_candidates[
                        wr_candidates["player_name"] == wr_name_sel
                    ].iloc[0]


        with col_db:
            if not db_candidates.empty:
                db_option_names = (
                    db_candidates.get("player_name", pd.Series(dtype=str))
                    .dropna()
                    .unique()
                    .tolist()
                )

                default_db_idx = 0
                if default_db_name_from_single and default_db_name_from_single in db_option_names:
                    default_db_idx = db_option_names.index(default_db_name_from_single)
                elif db_name and db_name in db_option_names:
                    default_db_idx = db_option_names.index(db_name)

                db_name_sel = st.selectbox(
                    "Focus defender in this play",
                    options=db_option_names,
                    index=default_db_idx,
                )

                focus_db_name = db_name_sel or focus_db_name

                if db_name_sel:
                    focus_db_row = db_candidates[
                        db_candidates["player_name"] == db_name_sel
                    ].iloc[0]


    # ---------- WR / DB FOCUS PANELS ----------
    left, right = st.columns(2)

    with left:
        st.markdown(
            f"""
<div class="play-panel">
  <div class="play-panel-title">
    Receiver focus – what this rep measures
  </div>
  <div class="play-panel-sub">
    {focus_wr_name or "Targeted receiver"} · offense
  </div>

  <div class="play-pill-row">
    <div class="play-pill-tag"><span class="code">A</span> Anticipation</div>
    <div class="play-pill-tag"><span class="code">S</span> Separation</div>
    <div class="play-pill-tag"><span class="code">E</span> Execution</div>
    <div class="play-pill-tag">Eyes</div>
    <div class="play-pill-tag">Innovation</div>
    <div class="play-pill-tag">Improv (I)</div>
  </div>

  <p class="play-panel-sub">
    For the receiver, this play is evaluated as:
    <strong>can he win the release, create usable space, track the ball cleanly,
    and still finish if the throw or leverage picture isn't perfect?</strong>
    Tracking data supplies the route path and separation curve; manual tags
    attach 1–10 scores for each pillar event on this rep.
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
<div class="play-panel">
  <div class="play-panel-title">
    Defender focus – what this rep measures
  </div>
  <div class="play-panel-sub">
    {focus_db_name or "Primary coverage"} · defense
  </div>

  <div class="play-pill-row">
    <div class="play-pill-tag"><span class="code">A</span> Anticipation</div>
    <div class="play-pill-tag"><span class="code">S</span> Space denial</div>
    <div class="play-pill-tag"><span class="code">E</span> Technique</div>
    <div class="play-pill-tag">Eyes</div>
    <div class="play-pill-tag">Innovation</div>
    <div class="play-pill-tag">Improv (I)</div>
  </div>

  <p class="play-panel-sub">
    For the DB, the same rep asks:
    <strong>does he time the jam, stay in phase through the vertical stem,
    locate the ball in time, and intelligently recover if the receiver wins
    leverage?</strong> His pillar tags on this play contribute to his press /
    vertical posterior in PER-10.
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Ball IQ matchup snapshot ----------
    try:
        render_matchup_scout_summary(focus_wr_row, focus_db_row)
    except Exception as e:
        st.caption(f"(Matchup snapshot unavailable: {e})")

    # ---------- TIMELINE + FEATURE BUNDLE ----------
    st.markdown("### How the rep is logged in the model")

    outcome_bits: list[str] = []
    if pass_result:
        pr_map = {
            "C": "completed",
            "I": "incomplete",
            "IN": "interception",
            "TD": "touchdown",
        }
        outcome_bits.append(pr_map.get(pass_result, pass_result))
    if not pd.isna(yards_gained):
        yg = int(yards_gained)
        outcome_bits.append(f"{yg} yards gained" if yg >= 0 else f"{abs(yg)} yards lost")
    outcome_text = " · ".join(outcome_bits) if outcome_bits else "Outcome not recorded"

    tcol1, tcol2 = st.columns([1.3, 1])

    with tcol1:
        st.markdown(
            f"""
<div class="play-panel">
  <div class="play-panel-title">
    Rep timeline · pillar events
  </div>

  <div class="play-timeline-step">
    <div class="play-timeline-label">
      1 · Pre-snap &amp; release · A, E
    </div>
    <div class="play-timeline-body">
      The model logs the first movement toward the eventual catch path as an
      <strong>A (Anticipation)</strong> event: does the receiver or DB trigger
      first given the coverage and leverage? Strike timing, feet and hand usage
      at the line are scored as <strong>E (Execution)</strong> on a 1–10 scale.
    </div>
  </div>

  <div class="play-timeline-step">
    <div class="play-timeline-label">
      2 · Stem &amp; separation · S, Innovation
    </div>
    <div class="play-timeline-body">
      Tracking points along the stem yield a separation curve: how far apart
      WR and DB are every few frames. Tempo changes, angle changes, or creative
      leverage moves are tagged as <strong>Innovation</strong>, and the
      resulting separation at key breakpoints feeds the <strong>S</strong> score.
    </div>
  </div>

  <div class="play-timeline-step">
    <div class="play-timeline-label">
      3 · Ball in the air · Eyes, A
    </div>
    <div class="play-timeline-body">
      Once the ball leaves the QB’s hand, the model tracks who finds it first.
      The first frame where a player’s tracking direction lines up with ball
      flight becomes an <strong>Eyes</strong> event, extending Anticipation
      credit deeper into the rep.
    </div>
  </div>

  <div class="play-timeline-step">
    <div class="play-timeline-label">
      4 · Catch point &amp; finish · E, S, Improv
    </div>
    <div class="play-timeline-body">
      At the arrival point, final separation, body control, and contest type
      are logged. If the route or throw goes off-script and either player
      creates a new answer on the fly, that decision is scored as
      <strong>Improv (I)</strong>. This play’s final label is:
      <strong>{outcome_text}</strong>.
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with tcol2:
        st.markdown(
            """
<div class="play-panel">
  <div class="play-panel-title">
    Feature bundle sent to PER-10 / PER-10 360
  </div>

  <p class="play-panel-sub">
    For this rep, Sky Vision records a structured feature bundle, for example:
  </p>

  <ul style="font-size:0.83rem; color:#cbd5f5; padding-left:1.1rem;">
    <li><strong>Game / play keys:</strong> gameId &amp; playId so we can link
        back to film and tracking.</li>
    <li><strong>Context:</strong> offense/defense, quarter, down &amp; distance,
        field position, route family, coverage family.</li>
    <li><strong>Tracking-derived features:</strong> separation over time,
        leverage (inside/outside/on-top), speed and acceleration at key
        timestamps.</li>
    <li><strong>Pillar events:</strong> 1–10 scores for Anticipation,
        Separation, Execution, Eyes, Innovation, and Improv on this rep.</li>
    <li><strong>Outcome:</strong> completion / incompletion / interception /
        TD, yards gained and YAC context.</li>
  </ul>

  <p class="play-footnote">
    Across many plays with the same route/coverage family, these per-rep
    feature bundles are pooled in a Bayesian model to produce the PER-10
    and PER-10 360 ratings you see on the other tabs. Selecting a different
    play above swaps in a completely different real feature bundle.
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )
