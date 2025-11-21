# scripts/05_compare.py
# streamlit app that adds in comparison mode

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# paths
ROOT       = Path(__file__).resolve().parents[1]
OUT_DIR    = ROOT / "outputs"
POST_DIR   = OUT_DIR / "posterior"
PILLARS_CSV = POST_DIR / "posterior_pillars.csv"
OVERALL_CSV = POST_DIR / "posterior_overall.csv"
CORE_CSV    = OUT_DIR / "merged_core.csv"

# pillar sets
WR_PILLARS = ["anticipation", "execution", "separation", "innovation", "eyes"]
DB_PILLARS = ["coverage", "reaction"]

# data downloading
@st.cache_data(show_spinner=True)
def load_all():
    if not (PILLARS_CSV.exists() and OVERALL_CSV.exists()):
        raise FileNotFoundError("Run 02_extract_pillars.py and 03_update_posterior.py first.")
    if not CORE_CSV.exists():
        raise FileNotFoundError("merged_core.csv not found. Run 01_merge_pipeline.py first.")

    pillars = pd.read_csv(PILLARS_CSV)
    overall = pd.read_csv(OVERALL_CSV)
    core    = pd.read_csv(CORE_CSV)

    # normalize columns
    for df in (pillars, overall, core):
        df.columns = [c.lower() for c in df.columns]

    # core has nfl_id, map to player_id for joining
    core = core.rename(columns={"nfl_id": "player_id"})

    # Consolidate roster info (unique per player & side/position when possible)
    roster = (core[['player_id', 'player_name', 'player_position', 'player_side']]
              .dropna(subset=['player_id'])
              .drop_duplicates())

    # Attach names/positions into overall & pillars
    overall = overall.merge(roster, on=['player_id'], how='left')
    pillars = pillars.merge(roster, on=['player_id'], how='left')

    # Build a "pool" for pickers
    pool = (overall[['player_id', 'player_name', 'side', 'player_position']]
            .dropna(subset=['player_id', 'player_name', 'side'])
            .drop_duplicates()
            .sort_values(['side', 'player_position', 'player_name']))

    # Clean up positions: fill NaN with "UNK"
    pool['player_position'] = pool['player_position'].fillna('UNK')

    return pillars, overall, pool

pillars, overall, pool = load_all()

# helpers
def pillar_order_for_side(side: str):
    return WR_PILLARS if side == "WR" else DB_PILLARS

def safe_player_options(side: str, position: str | None):
    df = pool[pool['side'] == side]
    if position and position != "All":
        df = df[df['player_position'].fillna('UNK') == position]
    return df

def safe_find_pid(_pool: pd.DataFrame, name: str) -> int | None:
    rows = _pool.loc[_pool['player_name'] == name, 'player_id']
    if rows.empty:
        return None
    return int(rows.iloc[0])

def scores_for_player(player_id: int, side: str) -> dict:
    """Return dict pillar -> score_1_10 (posterior mean*10 already)."""
    side_pillars = pillar_order_for_side(side)
    sub = pillars[(pillars['player_id'] == player_id) & (pillars['side'] == side)]
    # pivot mean over pillar in case there are multiple rows
    ser = (sub.groupby('pillar')['score_1_10'].mean() if not sub.empty else pd.Series(dtype=float))
    out = {p: (float(ser[p]) if p in ser.index else np.nan) for p in side_pillars}
    return out

def single_bar(scores: dict, side: str, title: str):
    order = pillar_order_for_side(side)
    y = [scores.get(p, np.nan) for p in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(order, y)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score (1–10)")
    ax.set_title(title)
    for i, v in enumerate(y):
        if not np.isnan(v):
            ax.text(i, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig)

def grouped_bars(scores_a: dict, scores_b: dict, side: str, name_a: str, name_b: str):
    order = pillar_order_for_side(side)
    A = [scores_a.get(p, np.nan) for p in order]
    B = [scores_b.get(p, np.nan) for p in order]

    x = np.arange(len(order))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, A, width=w, label=name_a)
    ax.bar(x + w/2, B, width=w, label=name_b)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score (1–10)")
    ax.set_title("Pillar Scores — Side-by-Side")
    ax.legend()

    # labels above bars
    for i, v in enumerate(A):
        if not np.isnan(v):
            ax.text(i - w/2, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(B):
        if not np.isnan(v):
            ax.text(i + w/2, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    st.pyplot(fig)

# UI 
st.set_page_config(page_title="Sky Vision — Compare", layout="wide")

with st.sidebar:
    st.header("Options")

    mode = st.radio("Mode", ["Player assessment", "Player comparison"], index=0)

    side_choice = st.radio("Side", ["WR", "DB"], index=0)

    # Position filter (values present in pool for this side)
    valid_pos = (pool[pool['side'] == side_choice]['player_position']
                 .dropna()
                 .replace('', 'UNK')
                 .sort_values()
                 .unique()
                 .tolist())
    positions = ["All"] + [p for p in valid_pos if p != 'UNK'] + (["UNK"] if "UNK" in valid_pos else [])
    pos_choice = st.selectbox("Position", positions, index=0)

st.title("Sky Vision")

# filtered pool once 
picker_pool = safe_player_options(side_choice, pos_choice)
if picker_pool.empty:
    st.warning("No players match the current filters.")
    st.stop()

if mode == "Player assessment":
    st.subheader("Player assessment")
    names = picker_pool['player_name'].tolist()
    pid_label = st.selectbox("Player", names, index=0)

    pid = safe_find_pid(picker_pool, pid_label)
    if pid is None:
        st.warning("Selected player not found in the filtered data.")
        st.stop()

    scores = scores_for_player(pid, side_choice)

    # Overall 0–100 (if present)
    ov = overall[(overall['player_id'] == pid) & (overall['side'] == side_choice)]
    if not ov.empty and 'overall_0_100' in ov.columns and pd.notna(ov['overall_0_100'].iloc[0]):
        st.metric("Overall (0–100)", f"{ov['overall_0_100'].iloc[0]:.1f}")

    # PER-10 if present
    if 'per10' in ov.columns and not ov.empty and pd.notna(ov['per10'].iloc[0]):
        st.metric("PER-10", f"{ov['per10'].iloc[0]:.1f}")

    single_bar(scores, side_choice, title=f"Pillar Scores (1–10) — {pid_label}")

else:
    st.subheader("Player comparison")

    # two independent pickers from the same filtered pool
    names = picker_pool['player_name'].tolist()
    col1, col2 = st.columns(2)
    with col1:
        pid_label_a = st.selectbox("Player A", names, index=0, key="a")
    with col2:
        pid_label_b = st.selectbox("Player B", names, index=min(1, len(names)-1), key="b")

    pid_a = safe_find_pid(picker_pool, pid_label_a)
    pid_b = safe_find_pid(picker_pool, pid_label_b)

    if pid_a is None or pid_b is None:
        st.warning("One or both selected players are not in the filtered pool.")
        st.stop()
    if pid_a == pid_b:
        st.warning("Please choose two different players.")
        st.stop()

    scores_a = scores_for_player(pid_a, side_choice)
    scores_b = scores_for_player(pid_b, side_choice)

    # Small summary table
    def pack(pid, name):
        ov = overall[(overall['player_id'] == pid) & (overall['side'] == side_choice)]
        ov100 = ov['overall_0_100'].iloc[0] if (not ov.empty and pd.notna(ov['overall_0_100'].iloc[0])) else np.nan
        per10 = ov['per10'].iloc[0]        if (not ov.empty and 'per10' in ov and pd.notna(ov['per10'].iloc[0])) else np.nan
        return {"player": name, "overall_0_100": ov100, "per10": per10}
    st.write(pd.DataFrame([pack(pid_a, pid_label_a), pack(pid_b, pid_label_b)]))

    grouped_bars(scores_a, scores_b, side_choice, pid_label_a, pid_label_b)
