
# 04_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST_DIR = ROOT / "outputs" / "posterior"
POST_PILL = POST_DIR / "posterior_pillars.csv"
POST_OVR  = POST_DIR / "posterior_overall.csv"

st.title("SKY VISION — Bayesian Pillars")

if not POST_PILL.exists() or not POST_OVR.exists():
    st.warning("Run 02_extract_pillars.py and 03_update_posterior.py first.")
    st.stop()

pillars = pd.read_csv(POST_PILL)
overall = pd.read_csv(POST_OVR)

# supports snake_case or legacy camelCase
rename = {}
if "playerId" in pillars.columns: rename["playerId"] = "player_id"
if "playerId" in overall.columns: rename["playerId"] = "player_id"
if rename:
    pillars = pillars.rename(columns=rename)
    overall = overall.rename(columns=rename)

player_ids = sorted(pillars["player_id"].unique())
pid = st.sidebar.selectbox("Player ID", player_ids)

# choose side (WR/DB) 
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

    # PER-10 and Quick Instinct 
    if "per10" in ov.columns and not ov.empty:
        per10_val = ov["per10"].iloc[0]
        if pd.notna(per10_val):
            st.metric("PER-10", f"{per10_val:.1f}")

            # Quick Instinct logic holds Eyes ≥ 8 and Innovation ≥ 8
            sub_player = pillars[(pillars["player_id"] == pid) & (pillars["side"] == side)]
            if not sub_player.empty:
                eyes_score = sub_player.loc[sub_player["pillar"] == "eyes", "score_1_10"]
                innov_score = sub_player.loc[sub_player["pillar"] == "innovation", "score_1_10"]

                eyes10  = eyes_score.iloc[0]  if not eyes_score.empty  else np.nan
                innov10 = innov_score.iloc[0] if not innov_score.empty else np.nan

                if pd.notna(eyes10) and pd.notna(innov10) and eyes10 >= 8 and innov10 >= 8:
                    st.success("Archetype: Quick Instinct")
    else:
        pass

with right:
    sub = pillars[(pillars["player_id"] == pid) & (pillars["side"] == side)].copy()
    if not sub.empty:
        sub = sub.sort_values("pillar")
        st.bar_chart(sub.set_index("pillar")["score_1_10"])
    else:
        st.write("No pillar data for this selection.")

st.write("Pillar Posteriors (α, β, mean)")
st.dataframe(pillars[(pillars["player_id"] == pid) & (pillars["side"] == side)])
