
# 04_dashboard.py

import streamlit as st
import pandas as pd
from pathlib import Path

POST_PILL = Path("../outputs/posterior/posterior_pillars.csv")
POST_OVR  = Path("../outputs/posterior/posterior_overall.csv")

st.title("SKY VISION — Bayesian Pillars")

if not POST_PILL.exists() or not POST_OVR.exists():
    st.warning("Run 02_extract_pillars.py and 03_update_posterior.py first.")
    st.stop()

pillars = pd.read_csv(POST_PILL)
overall = pd.read_csv(POST_OVR)

# Support either snake_case (current) or legacy camelCase
rename = {}
if "playerId" in pillars.columns: rename["playerId"] = "player_id"
if "playerId" in overall.columns: rename["playerId"] = "player_id"
if rename:
    pillars = pillars.rename(columns=rename)
    overall = overall.rename(columns=rename)

player_ids = sorted(pillars["player_id"].unique())
pid = st.sidebar.selectbox("Player ID", player_ids)

# Let the user choose side (WR/DB) if both exist
sides = sorted(pillars.loc[pillars["player_id"] == pid, "side"].unique())
side = st.sidebar.selectbox("Side", sides)

left, right = st.columns(2)
ov = overall[(overall["player_id"] == pid) & (overall["side"] == side)]

with left:
    st.subheader(f"Player {pid} — {side}")
    if not ov.empty and pd.notna(ov["overall_0_100"].iloc[0]):
        st.metric("Overall (0–100)", f"{ov['overall_0_100'].iloc[0]:.1f}")
    else:
        st.metric("Overall (0–100)", "—")

with right:
    sub = pillars[(pillars["player_id"] == pid) & (pillars["side"] == side)].copy()
    if not sub.empty:
        sub = sub.sort_values("pillar")
        st.bar_chart(sub.set_index("pillar")["score_1_10"])
    else:
        st.write("No pillar data for this selection.")

st.write("Pillar Posteriors (α, β, mean)")
st.dataframe(pillars[(pillars["player_id"] == pid) & (pillars["side"] == side)])
