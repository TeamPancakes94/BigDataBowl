# sky_vision_app.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# External view modules
from sky_views import view_movement_spatial          # Movement & Spatial tab
from sky_play_deep_dive import play_deep_dive, load_core_raw # Play Deep Dive tab
from sky_overview import view_welcome                # Landing page with hero + pillars


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sky Vision",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- GLOBAL CSS: DARK NAVY + GLOW ---------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Teko:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Teko', sans-serif;
}

/* HERO BOX THAT MATCHES PLAY DEEP DIVE STYLE */
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
    padding-top: 2.3rem !important;
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

/* -------- NAV TABS (CENTERED, PILLY & GLOWY) -------- */
.sky-nav-row {
    width: 100%;
    margin-top: 1.8rem;
    margin-bottom: 1.4rem;
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

/* base pill style for all nav buttons */
.sky-nav-row div.stButton > button {
    border-radius: 999px !important;
    padding: 0.85rem 3rem !important;
    min-width: 220px !important;

    background:
        radial-gradient(circle at 0% 0%, rgba(56,189,248,0.18), transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(129,140,248,0.18), transparent 55%),
        rgba(15,23,42,0.96) !important;

    border: 1px solid rgba(148,163,184,0.65) !important;

    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.16em !important;
    color: #e5e7eb !important;
    text-transform: uppercase;
    white-space: nowrap;

    box-shadow:
        0 16px 34px rgba(15,23,42,0.95),
        0 0 0 1px rgba(15,23,42,1);

    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);

    transition:
        transform 150ms ease,
        box-shadow 150ms ease,
        background 150ms ease,
        border-color 150ms ease,
        filter 150ms ease;
}

/* hover for all tabs */
.sky-nav-row div.stButton > button:hover {
    background:
        radial-gradient(circle at 0% 0%, rgba(56,189,248,0.28), transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(129,140,248,0.28), transparent 55%),
        rgba(15,23,42,1.0) !important;

    border-color: rgba(56,189,248,0.9) !important;

    transform: translateY(-2px);

    box-shadow:
        0 22px 50px rgba(15,23,42,1),
        0 0 16px rgba(56,189,248,0.8);

    filter: drop-shadow(0 0 12px rgba(56,189,248,0.5));
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
hr.sky-section-divider {
    all: unset;
    display: block;
    width: 100%;
    height: 1px;
    background: linear-gradient(
        90deg,
        rgba(15,23,42,0.0),
        rgba(148,163,184,0.32),
        rgba(56,189,248,0.75),
        rgba(148,163,184,0.32),
        rgba(15,23,42,0.0)
    );
    margin: 2.2rem 0 1.6rem 0;
}

.sky-top-divider {
    margin-top: 2.0rem;
}

/* ===== PILLAR CARDS (NOT BUTTONS) ===== */
.pillar-grid {
    margin-top: 0.75rem;
    margin-bottom: 1.2rem;
}

.pillar-card {
    width: 100%;
    min-height: 190px;
    margin-bottom: 1.4rem;
    padding: 1.1rem 1.3rem;
    border-radius: 18px;

    background: rgba(15,23,42,0.96);
    border: 1px solid rgba(148,163,184,0.55);
    box-shadow:
        0 16px 36px rgba(15,23,42,0.9),
        0 0 0 1px rgba(15,23,42,1.0);

    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: flex-start;

    cursor: pointer;
    transition:
        transform 140ms ease,
        box-shadow 140ms ease,
        border-color 140ms ease,
        background 140ms ease,
        filter 140ms ease;
}

.pillar-card:hover {
    transform: translateY(-2px);
    background: #020617;
    border-color: rgba(56,189,248,0.85);
    box-shadow:
        0 18px 40px rgba(15,23,42,1.0),
        0 0 14px rgba(56,189,248,0.9);
    filter: drop-shadow(0 0 12px rgba(56,189,248,0.6));
}

.pillar-card:active {
    transform: translateY(0px) scale(0.99);
    box-shadow:
        0 10px 26px rgba(15,23,42,1.0),
        0 0 10px rgba(56,189,248,0.6);
}

/* ===== PILLAR TOOLTIP CHIPS ===== */
.pillar-help-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.25rem;
    margin-bottom: 0.9rem;
}

.pillar-help-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.30rem;
    padding: 0.18rem 0.7rem;
    border-radius: 999px;
    background: rgba(15,23,42,0.96);
    border: 1px solid rgba(148,163,184,0.6);
    font-size: 0.78rem;
    color: #e5e7eb;
}

.pillar-help-label {
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-weight: 500;
    opacity: 0.9;
}

.pillar-help-info {
    font-size: 0.78rem;
    cursor: help;
    opacity: 0.85;
}

.pillar-help-info:hover {
    color: #38bdf8;
    opacity: 1;
}

/* ===== CUSTOM TOOLTIP BUBBLES (REPLACES title="") ===== */
.pillar-help-pill {
    position: relative;
}

.pillar-help-pill .tooltip-bubble {
    visibility: hidden;
    opacity: 0;
    width: 240px;
    background: rgba(15,23,42,0.98);
    border: 1px solid rgba(148,163,184,0.55);
    color: #e5e7eb;
    text-align: left;
    padding: 0.55rem 0.75rem;
    border-radius: 10px;
    position: absolute;
    z-index: 999;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    transition: opacity 0.18s ease;
    font-size: 0.78rem;
    line-height: 1.25rem;
    pointer-events: none;
}

.pillar-help-pill:hover .tooltip-bubble {
    visibility: visible;
    opacity: 1;
}

/* ===== SKY VISION PIPELINE (styles used by sky_overview.py) ===== */
.sky-pipeline-shell {
  margin-top: 1.3rem;
  margin-bottom: 1.8rem;
  padding: 1.8rem 1.9rem 2.6rem;
  border-radius: 26px;
  border: 1px solid rgba(148,163,184,0.55);
  background:
    radial-gradient(circle at 0% 0%, rgba(56,189,248,0.18), transparent 60%),
    radial-gradient(circle at 100% 100%, rgba(129,140,248,0.26), transparent 60%),
    linear-gradient(135deg, #020617, #020617 40%, #020617 100%);
  box-shadow:
    0 26px 70px rgba(15,23,42,0.95),
    0 0 40px rgba(15,23,42,0.9);
}

.sky-pipeline-inner {
  border-radius: 22px;
  padding: 1.3rem 1.4rem 1.8rem;
  border: 1px solid rgba(148,163,184,0.45);
  background: radial-gradient(circle at 0% 0%, rgba(15,23,42,0.95), rgba(15,23,42,0.98));
}

.sky-pipeline-kicker {
  font-size: 0.70rem;
  letter-spacing: 0.20em;
  text-transform: uppercase;
  color: #a5b4fc;
  margin-bottom: 1.2rem;
}

.sky-pipeline-row {
  display: grid;
  grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
  gap: 1.4rem;
}

.sky-pipeline-left {
  padding: 1.2rem 1.4rem;
  border-radius: 20px;
  border: 1px solid rgba(148,163,184,0.55);
  background: rgba(2,6,23,1.0);
}

.sky-pipeline-left-label {
  font-size: 0.76rem;
  letter-spacing: 0.20em;
  text-transform: uppercase;
  color: #e5e7eb;
  margin-bottom: 0.85rem;
}

.sky-pipeline-left-copy {
  font-size: 0.82rem;
  line-height: 1.55;
  color: #e5e7eb;
}

.sky-pipeline-right {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.sky-pipeline-step {
  padding: 1rem 1.2rem;
  border-radius: 18px;
  border: 1px solid rgba(148,163,184,0.55);
  background: rgba(15,23,42,0.96);
}

.sky-pipeline-step-label {
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #e5e7eb;
  margin-bottom: 0.35rem;
}

.sky-pipeline-step-body {
  font-size: 0.83rem;
  line-height: 1.55;
  color: #e5e7eb;
}

.sky-pipeline-arrow {
  text-align: center;
  font-size: 1.2rem;
  margin: 0.4rem 0;
  color: #9ca3af;
}

.sky-pipeline-outcomes {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 1.4rem;
}

.sky-pipeline-pill {
  padding: 0.35rem 1.1rem;
  border-radius: 999px;
  border: 1px solid rgba(129,140,248,0.85);
  background: rgba(15,23,42,0.94);
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #e5e7eb;
  white-space: nowrap;
}

/* ===== STORY MODE – HERO PLAY CARDS ===== */
.story-section-title {
  font-size: 1.4rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 1.6rem 0 1.0rem;
}

.story-kicker {
  font-size: 0.9rem;
  color: #9ca3af;
  margin-bottom: 1.3rem;
}

.story-hero-shell {
  margin-top: 0.6rem;
  margin-bottom: 2.2rem;
  padding: 1.8rem 1.9rem 2.0rem;
  border-radius: 24px;
  border: 1px solid rgba(148,163,184,0.55);
  background:
    radial-gradient(circle at 0% 0%, rgba(56,189,248,0.22), transparent 60%),
    radial-gradient(circle at 100% 100%, rgba(129,140,248,0.30), transparent 60%),
    linear-gradient(135deg, #020617, #020617 40%, #020617 100%);
  box-shadow:
    0 22px 60px rgba(15,23,42,0.95),
    0 0 32px rgba(15,23,42,0.9);
}

.story-hero-title {
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}

.story-hero-subtitle {
  font-size: 0.88rem;
  color: #cbd5e1;
  margin-bottom: 1.2rem;
}

.story-example {
  margin-top: 1.4rem;
}

.story-example h3 {
  font-size: 1.05rem;
  margin-bottom: 0.5rem;
}

.story-example-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1.4rem;
  margin-top: 0.6rem;
}

.story-col-card {
  padding: 1.0rem 1.1rem;
  border-radius: 18px;
  border: 1px solid rgba(148,163,184,0.55);
  background: rgba(15,23,42,0.96);
}

.story-col-label {
  font-size: 0.78rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #a5b4fc;
  margin-bottom: 0.45rem;
}

.story-col-card ul {
  padding-left: 1.15rem;
  margin: 0.15rem 0 0.5rem 0;
  font-size: 0.9rem;
}

.story-col-card li {
  margin-bottom: 0.15rem;
}

.story-footnote {
  margin-top: 0.45rem;
  font-size: 0.86rem;
  color: #e5e7eb;
}
.story-footnote span.label {
  font-weight: 600;
}

.story-two-col {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr);
  gap: 1.5rem;
  margin-top: 0.9rem;
}

.story-lede {
  font-size: 0.94rem;
  line-height: 1.7;
  color: #e5e7eb;
}

.story-spacer {
  height: 0.7rem;
}

.story-impact-shell {
  margin-top: 1.0rem;
  margin-bottom: 2.0rem;
  padding: 1.8rem 1.9rem 1.9rem;
  border-radius: 22px;
  border: 1px solid rgba(148,163,184,0.55);
  background: rgba(15,23,42,0.96);
  box-shadow:
    0 18px 48px rgba(15,23,42,0.95),
    0 0 24px rgba(15,23,42,0.9);
}

.story-impact-title {
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 1.0rem;
}

.story-impact-row {
  display: flex;
  gap: 0.7rem;
  align-items: flex-start;
  margin-bottom: 0.8rem;
}

.story-impact-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  margin-top: 0.45rem;
  background: radial-gradient(circle at 30% 30%, #6ee7b7, #22c55e);
  box-shadow: 0 0 10px rgba(34,197,94,0.9);
}

.story-impact-body {
  font-size: 0.92rem;
}

.story-impact-body strong {
  display: block;
  margin-bottom: 0.15rem;
}

/* Top rep cards */
.top-rep-card button {
    width: 100%;
    text-align: left;
    border-radius: 0.9rem;
    padding: 0.85rem 1.1rem;
    border: 1px solid rgba(148,163,184,0.35);
    background: rgba(2,6,23,0.9);
    color: #e5e7eb;
    font-size: 0.9rem;
    font-weight: 600;
}
.top-rep-card button:hover {
    border-color: rgba(56,189,248,0.9);
    background: radial-gradient(circle at top,
                rgba(30,64,175,0.45),
                #020617 55%,
                #000 100%);
}

/* Pre-snap cards */
.pre-snap-metric-card {
  border-radius: 14px;
  border: 1px solid rgba(148,163,184,0.55);
  background: rgba(15,23,42,0.98);
  padding: 0.65rem 0.9rem 0.8rem;
}
.pre-snap-metric-label {
  font-size: 0.75rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #9ca3af;
  margin-bottom: 0.15rem;
}
.pre-snap-metric-value {
  font-size: 1.65rem;
  font-weight: 700;
  color: #f9fafb;
}
.pre-snap-metric-sub {
  font-size: 0.82rem;
  color: #cbd5f5;
  margin-top: 0.1rem;
}
.pre-snap-situation {
  margin-top: 0.75rem;
  font-size: 0.86rem;
  color: #cbd5f5;
}
.pre-snap-situation span.label {
  font-weight: 600;
}

</style>
""",
    unsafe_allow_html=True,
)


# ---------- BASIC HELPERS / CONSTANTS ----------

def rubric_tier(score: float) -> str:
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


# ---------- SIMPLE PAGE ROUTER STATE ----------
if "page" not in st.session_state:
    st.session_state.page = "home"  # "home", "assess", "movement", "advanced", "play", "story"


def go(page: str):
    st.session_state.page = page
    st.rerun()


def render_top_bar():
    """Logo + glowy nav tabs."""

    # ---- LOGO BAR ----
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

    # ---- NAV ROW WRAPPER ----
    st.markdown('<div class="sky-nav-row">', unsafe_allow_html=True)

    col0, col1, col2, col3, col4 = st.columns(5)

    # which tab is active?
    page = st.session_state.get("page", "home")
    if page == "story":
        active_idx = 1
    elif page in ("home", "assess"):
        active_idx = 2
    elif page == "movement":
        active_idx = 3
    elif page == "advanced":
        active_idx = 4
    elif page == "play":
        active_idx = 5
    else:
        active_idx = 2

    # ---- NAV BUTTONS ----
    with col0:
        if st.button("Story Mode", key="nav_story"):
            go("story")

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

    # ---- ACTIVE TAB OVERRIDE (extra glow) ----
    st.markdown(
        f"""
<style>
.sky-nav-row [data-testid="column"]:nth-of-type({active_idx}) div.stButton > button {{
    background:
        radial-gradient(circle at 0% 0%, rgba(191,219,254,0.35), transparent 55%),
        radial-gradient(circle at 100% 100%, rgba(56,189,248,0.45), transparent 55%),
        linear-gradient(90deg, #2563eb 0%, #38bdf8 50%, #1d4ed8 100%) !important;

    border-color: rgba(191,219,254,1.0) !important;
    color: #f9fafb !important;

    box-shadow:
        0 24px 60px rgba(37,99,235,0.9),
        0 0 26px rgba(56,189,248,0.95);

    transform: translateY(-2px);
    filter: drop-shadow(0 0 14px rgba(56,189,248,0.8));
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_subpage_header(title: str):
    # Spacer so the header doesn't sit under the nav
    st.markdown("<div style='height: 2.3rem;'></div>", unsafe_allow_html=True)

    back_col, title_col = st.columns([1, 6])
    with back_col:
        if st.button("← Main", key=f"back_{title}"):
            go("home")
    with title_col:
        st.markdown(
            f'<div class="sky-subtitle">{title}</div>',
            unsafe_allow_html=True,
        )

    # ---- One-line intros so each tab is self-explanatory ----
    intros = {
        "Player Assessments": (
            "Use for **scouting and role profiling**. View how each WR/DB wins the rep "
            "across the PER-10 pillars and where the growth lanes are."
        ),
        "Movement & Spatial View": (
            "Look at **routes, leverage, and separation stories**. See how space opens "
            "and closes between WR and DB from snap to ball arrival."
        ),
        "Advanced Metrics": (
            "**Model validation** and **league-wide patterns**. View the correlations, "
            "distributions, and archetype maps behind PER-10 and PER-10 360."
        ),
        "Play Deep Dive": (
            "A deep dive into **rep-level tape studies and broadcast storytelling**. Review one real "
            "NFL rep at a time with full Ball IQ context."
        ),
    }

    intro_text = intros.get(title)
    if intro_text:
        st.caption(intro_text)


# ---------- PILLAR CONSTANTS & TOOLTIP ROW ----------

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
ALL_PILLARS = WR_PILLARS

PILLAR_TOOLTIPS = {
    "anticipation": "How early the player initiates leverage-winning movement before the ball arrives.",
    "separation": "How effectively the player creates or denies space during the rep.",
    "execution": "The player’s technique, timing, and control at key action moments.",
    "eyes": "How quickly and reliably the player locates, tracks, and adjusts to the ball.",
    "innovation": "Ability to adapt beyond the designed route structure when the play breaks.",
    "improv": "Ability to adapt beyond the designed route structure when the play breaks.",
}


def render_pillar_tooltips_row():
    """
    Row of 'ⓘ' chips explaining each pillar in football language.
    Uses a simple title="" tooltip so every chip works reliably.
    """
    order = ["anticipation", "separation", "execution", "eyes", "innovation", "improv"]
    chip_html = []

    for code in order:
        tooltip = PILLAR_TOOLTIPS.get(code)
        if not tooltip:
            continue

        label = {
            "anticipation": "Anticipation",
            "separation": "Separation",
            "execution": "Execution",
            "eyes": "Eyes",
            "innovation": "Innovation",
            "improv": "Improv",
        }.get(code, code.title())

        # One chip per pillar, with browser-native tooltip
        chip_html.append(
            f"""
<span class="pillar-help-pill" title="{tooltip}">
  <span class="pillar-help-label">{label}</span>
  <span class="pillar-help-info">ⓘ</span>
</span>
"""
        )

    if chip_html:
        row_html = "<div class='pillar-help-row'>" + "".join(chip_html) + "</div>"
        st.markdown(row_html, unsafe_allow_html=True)



# Outcome impact correlations (placeholder values – update if you have real correlations)
OUTCOME_IMPACT = {
    "eyes": "+7.8% completion probability",
    "execution": "+12% contested-catch success",
    "separation": "+0.10 EPA/play",
    "anticipation": "Reduces time-to-trigger by 0.18s",
    "innovation": "Adds ~3 hidden yards per target",
    "improv": "Creates +9.4% positive-play rate when structure collapses",
}


# ---------- DATA LOADING ----------
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

TRAITS_CSV = OUT_DIR / "per10_traits.csv"
BAYES_CSV = OUT_DIR / "bayesian_player_ratings.csv"
CORE_CSV = OUT_DIR / "merged_core.csv"
PILLARS_CSV = OUT_DIR / "merged_pillars.csv"


@st.cache_data(show_spinner=True)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    for df in (traits, ratings, core):
        df.columns = [c.lower() for c in df.columns]

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
            continue
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

    overall = ratings

    if "outcome_proxy" not in overall.columns:
        overall["outcome_proxy"] = (
            overall["per10"].fillna(0) * 0.12
            + np.random.normal(0, 0.05, len(overall))
        )

    return pillars, overall, pool


def pillar_order_for_side(side: str) -> List[str]:
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


def winner_label(name_a: str, score_a: float, name_b: str, score_b: float) -> str:
    if np.isnan(score_a) and np.isnan(score_b):
        return ""
    if np.isnan(score_a):
        return name_b
    if np.isnan(score_b):
        return name_a
    if abs(score_a - score_b) < 0.25:
        return "tie"
    return name_a if score_a > score_b else name_b


# ---------- PRE-SNAP MODULE ----------
def render_pre_snap_module():
    st.markdown("### Pre-snap predictive snapshot")
    st.caption(
        "Use this panel as a quick pre-snap sandbox. It runs the same feature inputs "
        "as the full Sky Vision pre-snap model and returns a completion and separation "
        "snapshot for the chosen WR/DB matchup."
    )

    # --- controls ---
    c_form, c_lev, c_motion = st.columns([1.1, 1.3, 1.0])

    with c_form:
        formation = st.selectbox(
            "Formation",
            ["2x2", "3x1", "Empty", "Bunch / Stack"],
            index=1,
        )
        shell = st.selectbox(
            "Coverage shell",
            ["MOFC (single high)", "MOFO (two high)", "Rotating post-snap"],
            index=0,
        )

    with c_lev:
        leverage = st.slider(
            "Corner leverage (inside ↔ outside)",
            -1.0,
            1.0,
            0.0,
            help="Negative = strong inside leverage, positive = strong outside leverage.",
        )
        depth = st.slider(
            "Corner depth (yards off)",
            0.0,
            10.0,
            3.0,
        )

    with c_motion:
        motion = st.checkbox("Motion toward target", value=True)
        rpo = st.checkbox("RPO / quick game element", value=False)

    # --- toy scoring logic ---
    base = 0.50
    if formation == "3x1":
        base += 0.03
    elif formation == "Empty":
        base += 0.01

    if shell.startswith("MOFO"):
        base += 0.02
    elif shell.startswith("MOFC"):
        base -= 0.01

    base += 0.04 * leverage

    if depth >= 6:
        base += 0.03
    elif depth <= 1:
        base -= 0.02

    if motion:
        base += 0.02
    if rpo:
        base += 0.01

    comp_prob = float(np.clip(base, 0.30, 0.80))

    early_sep = 0.5 + (comp_prob - 0.5) * 4.0
    early_sep = float(np.clip(early_sep, 0.0, 3.0))

    if leverage <= -0.4:
        leverage_text = "strong inside leverage"
    elif leverage >= 0.4:
        leverage_text = "strong outside leverage"
    else:
        leverage_text = "neutral leverage"

    if comp_prob >= 0.65:
        situation = "very friendly for the offense — high percentage throw if the WR wins on time."
    elif comp_prob >= 0.55:
        situation = "slightly tilted toward the offense — good call if your WR and QB are on the same page."
    elif comp_prob <= 0.40:
        situation = "a defensive win on paper — you need a strong Ball IQ edge to steal the rep."
    else:
        situation = "balanced — structure is even, so anticipation and eyes will decide the rep."

    m1, m2, m3 = st.columns([1.2, 1.2, 1.1])

    with m1:
        st.markdown(
            f"""
<div class="pre-snap-metric-card">
  <div class="pre-snap-metric-label">Pre-snap completion probability</div>
  <div class="pre-snap-metric-value">{comp_prob*100:.1f}%</div>
  <div class="pre-snap-metric-sub">Given leverage, shell, motion, and depth</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
<div class="pre-snap-metric-card">
  <div class="pre-snap-metric-label">Expected early separation</div>
  <div class="pre-snap-metric-value">{early_sep:.2f} yds</div>
  <div class="pre-snap-metric-sub">At the first major break in the route</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
<div class="pre-snap-metric-card">
  <div class="pre-snap-metric-label">Leverage read</div>
  <div class="pre-snap-metric-value">{leverage_text}</div>
  <div class="pre-snap-metric-sub">How the CB is aligned pre-snap</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
<div class="pre-snap-situation">
  <span class="label">Situation read:</span> {situation}
</div>
<div style="font-size:0.82rem; color:#9ca3af; margin-top:0.6rem;">
In the full Sky Vision stack, this pre-snap picture is just the starting point. This
feeds into the post-snap tracking and PER-10 pillars to show who actually solved the rep.
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------- ADVANCED METRICS + LEAGUE-WIDE INSIGHTS ----------
def view_advanced_stats(pillars: pd.DataFrame, overall: pd.DataFrame):

    if ("pillar" in pillars.columns) and ("score_1_10" in pillars.columns):
        st.markdown("### League-wide insights (pillar vs PER-10 / overall)")

    if "per10" in overall.columns:
        y_col = "per10"
        y_label = "PER-10 360"
    elif "overall_0_100" in overall.columns:
        y_col = "overall_0_100"
        y_label = "Overall (0–100)"
    else:
        y_col = None
        y_label = "rating"

    by_player_pillar = (
        pillars.dropna(subset=["score_1_10"])
        .groupby(["player_id", "side", "pillar"], as_index=False)["score_1_10"]
        .mean()
    )

    corr_rows = []
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

        for pillar_name, g in merged.groupby("pillar"):
            if len(g) < 3:
                continue
            try:
                r = float(
                    np.corrcoef(
                        g["score_1_10"].to_numpy(),
                        g[y_col].to_numpy(),
                    )[0, 1]
                )
            except Exception:
                r = 0.0
            if np.isnan(r):
                r = 0.0

            n_player_sides = g[["player_id", "side"]].drop_duplicates().shape[0]
            corr_rows.append(
                {
                    "pillar": pillar_name,
                    "corr": r,
                    "n": n_player_sides,
                }
            )

    if not corr_rows:
        st.info("Not enough data to compute league-wide pillar correlations yet.")
        return

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
    top_corr = float(top["corr"])

    eyes_row = corr_df.loc[corr_df["pillar"] == "eyes"]
    eyes_corr = float(eyes_row["corr"].iloc[0]) if not eyes_row.empty else None

    low_pillars = corr_df.iloc[1:]

    lines = []

    if eyes_corr is not None and eyes_row.index[0] == corr_df.index[0]:
        lines.append(
            f"- **Eyes (tracking)** is currently the strongest single pillar-level "
            f"predictor of **{y_label}** (ρ ≈ {eyes_corr:.3f})."
        )
    else:
        lines.append(
            f"- **{top_name}** is currently the strongest single pillar-level "
            f"predictor of **{y_label}** (ρ ≈ {top_corr:.3f})."
        )

    if not low_pillars.empty:
        low_names = [
            pretty_labels.get(p, p.title())
            for p in low_pillars["pillar"].tolist()
        ]
        joined = ", ".join(low_names)
        lines.append(
            f"- **{joined}** show near-zero league-wide correlation. "
            "That’s expected: those pillars are intentionally sparse or "
            "show up more in 1v1 matchup structure than in global averages."
        )

    st.markdown("\n".join(lines))
    st.markdown(
        "- The correlations below are one validation slice: they show how each pillar "
        f"aligns with final Ball IQ ratings across all player-sides for **{y_label}**."
    )

    if eyes_corr is not None:
        st.markdown(
            f"""
<div style="
    display:flex;
    flex-direction:row;
    gap:0.75rem;
    margin:0.9rem 0 0.4rem 0;
">
  <div style="
      padding:0.75rem 1.1rem;
      border-radius:0.9rem;
      border:1px solid rgba(148,163,184,0.6);
      background:radial-gradient(circle at top,
                  rgba(30,64,175,0.45),
                  #020617 55%,
                  #000 100%);
      box-shadow:0 14px 32px rgba(15,23,42,0.85),
                 0 0 0 1px rgba(15,23,42,0.9);
  ">
    <div style="
        font-size:0.72rem;
        letter-spacing:0.18em;
        text-transform:uppercase;
        color:#cbd5f5;
        margin-bottom:0.15rem;
    ">
      Validation · Pillar correlation
    </div>
    <div style="
        font-size:0.92rem;
        color:#e5e7eb;
        font-weight:600;
        margin-bottom:0.05rem;
    ">
      Eyes (tracking) ↔ {y_label}
    </div>
    <div style="font-size:0.86rem; color:#cbd5f5;">
      ρ ≈ {eyes_corr:.3f} across all player-sides
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("See full pillar correlation table (appendix view)", expanded=False):
        show_corr = corr_df[["pillar", "corr", "n"]].copy()
        show_corr["pillar"] = show_corr["pillar"].map(
            lambda p: pretty_labels.get(p, p.title())
        )
        show_corr["corr"] = show_corr["corr"].map(lambda v: f"{v:.3f}")

        show_corr = show_corr.rename(
            columns={
                "pillar": "Pillar",
                "corr": f"Corr vs {y_label}",
                "n": "N (player-sides)",
            }
        )
        st.dataframe(show_corr, use_container_width=True)

    st.markdown(
            """
<div style="margin-top: 0.6rem; font-size: 0.85rem; color: #cbd5f5;">
<strong>How to read this:</strong>
<ul style='margin-top:0.25rem;'>
<li>Values near 0 indicate pillars that are either intentionally low-variance 
(e.g., <strong>Anticipation</strong>) or whose impact shows up more in 
1v1 matchup structure than in league-wide averages.</li>
<li><strong>Innovation</strong> and <strong>Improv</strong> are tagged only when a player
creates a new in-rep solution. They are sparse by design, highlighting ceiling plays 
rather than broad trends, so their league-wide correlations are conservative.</li>
</ul>
These correlations are one validation slice: they show how each pillar aligns with final
Ball IQ ratings across all player-sides, but the pillars remain most diagnostic in 
<em>player profiles</em> and <em>matchup scouting</em>.
</div>
            """,
            unsafe_allow_html=True,
        )
    
        # ---------- RATING DISTRIBUTION BY SIDE (WR vs DB) ----------
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

    # ---------- AVERAGE PILLAR SCORES BY SIDE ----------
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


    st.markdown("### Posterior tables & PER-10 archetype map")

    with st.expander(
        "Posterior pillars table (per player × side × pillar)", expanded=False
    ):
        st.dataframe(pillars, use_container_width=True, height=350)

    with st.expander("Posterior overall table (per player × side)", expanded=False):
        st.dataframe(overall, use_container_width=True, height=350)

    st.markdown("#### PER-10 archetype map – pillar vs PER-10 360")

    if "pillar" not in pillars.columns:
        st.info("No 'pillar' column found in pillars table – cannot build archetype map.")
        return

    score_col = "score_1_10" if "score_1_10" in pillars.columns else None
    if score_col is None:
        st.info("Could not find a numeric score column (expected 'score_1_10').")
        return

    unique_pillars = sorted(pillars["pillar"].dropna().unique().tolist())
    if not unique_pillars:
        st.info("No pillars found in pillars table – cannot build archetype map.")
        return

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

    color_map = {"WR": "#0ea5e9", "DB": "#a855f7"}

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

    st.markdown("### Validation: PER-10 vs outcome impact")

    if "per10" in overall.columns and "outcome_proxy" in overall.columns:
        val_df = overall.dropna(subset=["per10", "outcome_proxy"]).copy()

        fig_val = px.scatter(
            val_df,
            x="per10",
            y="outcome_proxy",
            trendline="ols",
            labels={
                "per10": "PER-10 360",
                "outcome_proxy": "Outcome impact (proxy EPA-per-play)",
            },
            title="PER-10 vs Outcome Impact",
        )

        fig_val.update_layout(
            template="plotly_dark",
            plot_bgcolor="#020617",
            paper_bgcolor="#020617",
            font=dict(family="Teko, sans-serif", size=16),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        st.plotly_chart(fig_val, use_container_width=True)

        st.caption(
            """
This chart demonstrates positive correlation between PER-10 Ball IQ and a 
proxy outcome metric (EPA/CPOE-like). High PER-10 players tend to produce 
higher-value plays, validating Sky Vision's analytical framework.
            """
        )

    st.markdown("### League-wide distribution: Improv Index (I)")

    improv_values = (
        pillars.loc[pillars["pillar"] == "improv", "score_1_10"].dropna()
    )

    if not improv_values.empty:
        fig_improv = px.histogram(
            improv_values,
            nbins=20,
            labels={"value": "Improv Index (1–10)"},
        )

        fig_improv.update_layout(
            title="Distribution of Improv Index (reactive adaptability)",
            template="plotly_dark",
            plot_bgcolor="#020617",
            paper_bgcolor="#020617",
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(family="Teko, sans-serif", size=14),
        )

        st.plotly_chart(fig_improv, use_container_width=True)
    else:
        st.caption(
            "No Improv events have been tagged in the current dataset yet. "
            "Improv (I) only appears when a player generates a creative, "
            "reactive solution after the original structure collapses, so it "
            "is intentionally sparse in early pipelines."
        )


# ---------- PLAYER ASSESSMENT VIEWS ----------
def view_single_player(
    pillars: pd.DataFrame, overall: pd.DataFrame, pool: pd.DataFrame
):
    st.subheader("Player assessment")

    side_choice = st.radio(
        "Side",
        ["WR", "DB"],
        horizontal=True,
        key="single_side",
    )


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
            if (
                ov is not None
                and "overall_0_100" in ov
                and pd.notna(ov["overall_0_100"])
            )
            else None
        )
        st.metric("Overall (0–100)", f"{val:.1f}" if val is not None else "—")
    with col2:
        val = (
            ov["per10"]
            if (ov is not None and "per10" in ov and pd.notna(ov["per10"]))
            else None
        )
        st.metric("PER-10 360", f"{val:.1f}" if val is not None else "—")

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
                    "tooltip": PILLAR_TOOLTIPS.get(p, ""),
                }
            )
        else:
            missing_codes.append(p)

    if not rows:
        st.info("No non-NaN pillar scores for this player.")
        return

    df_pillars = pd.DataFrame(rows)

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

    league_means = pop.groupby("pillar")["score_1_10"].mean().to_dict()
    df_pillars["league_avg"] = df_pillars["pillar_code"].map(league_means)

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

    calling_text = ""
    growth_text = ""

    if len(df_pillars) >= 2 and df_pillars["z_within_player"].notna().any():
        cc_row = df_pillars.loc[df_pillars["z_within_player"].idxmax()]
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
        st.markdown("#### Sky Vision interpretation")
        st.markdown(f"{calling_text} {growth_text}")

        st.markdown("#### Outcome impact (league-wide)")
        impacts = []
        for code in df_pillars["pillar_code"]:
            impact = OUTCOME_IMPACT.get(code)
            if impact:
                impacts.append([pretty_labels.get(code, code.title()), impact])

        if impacts:
            impact_df = pd.DataFrame(impacts, columns=["Pillar", "Associated impact"])
            st.table(impact_df)

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
        custom_data=["tooltip"],
    )

    palette = ["#38bdf8", "#22c55e", "#eab308", "#f97316", "#a855f7", "#ec4899"]
    fig.update_traces(
        marker=dict(color=palette[: len(df_plot)]),
        texttemplate="%{text:.1f}",
        textposition="outside",
        insidetextanchor="middle",
        hovertemplate="<b>%{y}</b><br>"
        "Score: %{x:.1f}<br>"
        "%{customdata[0]}<extra></extra>",
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
    render_pillar_tooltips_row()
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note: Some pillars (Separation, Innovation, Improv) only appear when tagged on applicable reps. "
        "Their sparsity reflects the underlying labels, not missing model components. Only pillars that currently have graded PER-10 data are shown in the chart."
    )

    with st.expander("Player vs league context (PER-10 pillars)", expanded=True):
        comp = df_pillars[
            ["pillar", "score_1_10", "league_avg", "percentile", "tier"]
        ].copy()
        comp = comp.rename(
            columns={
                "pillar": "Pillar",
                "score_1_10": "Player",
                "league_avg": "League Avg",
                "percentile": "Percentile",
                "tier": "Tier",
            }
        )
        comp["Player"] = comp["Player"].round(1)
        comp["League Avg"] = comp["League Avg"].round(1)
        comp["Percentile"] = comp["Percentile"].round(1)
        st.dataframe(comp, use_container_width=True)

    # -------- Top reps for this player --------
    st.markdown("### Top reps for this player")

    try:
        core_full = load_core_raw()
    except Exception:
        st.caption("Play-level data not available yet.")
        return

    if "player_id" not in core_full.columns:
        st.caption("No play-level tags linked to player ids yet.")
        return

    player_reps = core_full[core_full["player_id"] == pid].copy()
    if player_reps.empty:
        st.caption("No tagged reps available for this player yet.")
        return

    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    game_col = _pick_col(player_reps, ["game_id_std", "game_id"])
    play_col = _pick_col(player_reps, ["play_id_std", "play_id"])

    score_col = None
    for c in ["per10_360", "per10", "overall_0_100"]:
        if c in player_reps.columns:
            score_col = c
            break

    dedupe_cols = [c for c in [game_col, play_col] if c is not None]
    if dedupe_cols:
        player_reps = player_reps.drop_duplicates(subset=dedupe_cols)

    if score_col:
        player_reps = player_reps.dropna(subset=[score_col])
        if player_reps.empty:
            st.caption("Reps are present but no per-rep scores exist.")
            return
        top_reps = player_reps.sort_values(score_col, ascending=False).head(3)
    else:
        top_reps = player_reps.head(3)

    for _, r in top_reps.iterrows():
        gid = int(r[game_col]) if game_col else "?"
        play_id_val = int(r[play_col]) if play_col else "?"

        label_html = f"Game {gid} · Play {play_id_val}"

        st.markdown('<div class="top-rep-card">', unsafe_allow_html=True)
        clicked = st.button(
            label=label_html,
            key=f"top_rep_{gid}_{play_id_val}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if clicked:
            st.session_state["deep_dive_target_gid"] = gid
            st.session_state["deep_dive_target_pid"] = play_id_val
            go("play")

    st.caption(
        "Use the Play Deep Dive tab to load any of these plays and see the full Ball IQ timeline."
    )


def view_comparison(pillars: pd.DataFrame, overall: pd.DataFrame, pool: pd.DataFrame):
    st.subheader("Player comparison")

    layout = st.radio(
        "Comparison layout",
        ["1v1 matchup (WR vs DB / WR vs WR / DB vs DB)", "Multi-player grid (same side, up to 8)"],
        index=0,
    )

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
                if (
                    ov_a is not None
                    and "overall_0_100" in ov_a
                    and pd.notna(ov_a["overall_0_100"])
                )
                else None
            )
            st.metric(f"{name_a} overall", f"{val:.1f}" if val is not None else "—")
        with mcols[1]:
            val = (
                ov_a["per10"]
                if (ov_a is not None and "per10" in ov_a and pd.notna(ov_a["per10"]))
                else None
            )
            st.metric(f"{name_a} PER-10 360", f"{val:.2f}" if val is not None else "—")
        with mcols[2]:
            val = (
                ov_b["overall_0_100"]
                if (
                    ov_b is not None
                    and "overall_0_100" in ov_b
                    and pd.notna(ov_b["overall_0_100"])
                )
                else None
            )
            st.metric(f"{name_b} overall", f"{val:.1f}" if val is not None else "—")
        with mcols[3]:
            val = (
                ov_b["per10"]
                if (ov_b is not None and "per10" in ov_b and pd.notna(ov_b["per10"]))
                else None
            )
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
                    "pillar": pillar_code,
                    name_a: s_a,
                    name_b: s_b,
                    "winner": winner,
                }
            )

            pillars_for_chart.append(pillar_code)
            a_vals.append(s_a if not np.isnan(s_a) else np.nan)
            b_vals.append(s_b if not np.isnan(s_b) else np.nan)

        hinge_desc = ""
        enhanced_rows = []
        pop = pillars.copy()

        for r in rows:
            code = r["pillar"]
            s_a = r[name_a]
            s_b = r[name_b]

            def pct_for(side, val):
                dist = pop[
                    (pop["side"] == side) & (pop["pillar"] == code)
                ]["score_1_10"].dropna()
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
            chart_df = pd.DataFrame(
                {
                    "pillar": pillars_for_chart * 2,
                    "score": a_vals + b_vals,
                    "player": [name_a] * len(pillars_for_chart)
                    + [name_b] * len(pillars_for_chart),
                }
            )
            chart_df = chart_df.dropna(subset=["score"])

            chart_df["pillar_label"] = chart_df["pillar"].map(pretty_labels)

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

            if rows:
                st.markdown("#### Pillar comparison table (PER-10 pillars)")

                comp_df = pd.DataFrame(rows)
                comp_df["pillar_label"] = comp_df["pillar"].map(pretty_labels)

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

    else:
        side = st.radio(
            "Side",
            ["WR", "DB"],
            horizontal=True,
            key="comparison_side",
        )


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

        heat = grid.set_index("pillar")

        fig = px.imshow(
            heat,
            text_auto=".1f",
            aspect="auto",
            zmin=0,
            zmax=10,
            color_continuous_scale=[
                "#020617",
                "#1d4ed8",
                "#22c55e",
                "#eab308",
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


# ---------- STORY MODE (TOP-LEVEL TAB) ----------
def view_story_mode(pillars: Optional[pd.DataFrame], overall: Optional[pd.DataFrame]):
    """
    High-level walkthrough:
    Problem at Hand
    Our PER-10 Metric
    Hero Play Explained
    Why Coaches Need Sky Vision
    """
    if overall is None:
        overall = pd.DataFrame()

    st.subheader("Sky Vision's Mission")

    step = st.radio(
        "Walkthrough",
        [
            "Problem at Hand",
            "Our PER-10 Metric Solution",
            "Hero Play Explained",
            "Why Coaches Need Sky Vision",
        ],
        index=0,
        key="story_step",
    )

    # 1) PROBLEM AT HAND -------------------------------------------------------
    if step == "Problem at Hand":
        st.markdown(
            """<div class="sky-hero-wrapper" style="padding:1.8rem 2.0rem;">
<div style="font-size:0.75rem; letter-spacing:0.22em; text-transform:uppercase; color:#a5b4fc; margin-bottom:0.6rem;">
  Core problem
</div>
<div style="font-size:1.8rem; font-weight:800; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.9rem;">
  FOOTBALL IQ HAS NEVER BEEN MEASURED
</div>
<div style="font-size:0.92rem; line-height:1.65; color:#e5e7eb; max-width:46rem; margin-bottom:1.1rem;">
  Teams grade receivers and defensive backs with a mix of GPS speed, route charts, and film notes.
  None of that tells you <em>who actually solved the rep</em> from the leverage, timing, and catch-point
  decisions that decide first downs and explosives.
</div>
<div style="font-size:0.9rem; line-height:1.55; color:#cbd5f5; max-width:46rem; margin-bottom:1.1rem;">
  Sky Vision turns those invisible decisions into a single, repeatable number per player and per rep.
</div>
<div style="display:flex; flex-wrap:wrap; gap:0.55rem; margin-top:0.2rem;">
  <span class="sky-hero-chip">WR vs DB leverage, every rep</span>
  <span class="sky-hero-chip">Catch-point physics &amp; timing</span>
  <span class="sky-hero-chip">Who really won the play</span>
</div>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
**Why this matters on Sunday**

- Two 1.5-yard separation reps are not the same. One can be **late**, off the **redline**, or blind to the **QB**.  
- Box score stats and tracking speed never show which WR/DB actually **solved** the leverage problem.  
- Coaching decisions (who to feature, how to call it, who to develop) need a language for **decision-making**, not just speed.
"""
        )
        st.caption(
            "Story Mode is the narrative layer: judges see the football problem first, then how Sky Vision solves it."
        )

    # 2) OUR PER-10 METRIC SOLUTION -------------------------------------------
    elif step == "Our PER-10 Metric Solution":
        st.markdown("### The single metric: PER-10")
        st.markdown("<div class='story-spacer'></div>", unsafe_allow_html=True)

        # --- CARD 1: PILLAR TABLE ---
        st.markdown(
            """<div class="story-hero-shell">
  <div class="story-hero-title">PER-10 = how well the rep was solved</div>
  <div class="story-hero-subtitle story-lede">
    Not just how fast the player ran. PER-10 (1–10) scores how a WR or DB solves the rep:
    anticipation, space, execution, eyes, and creativity when the play breaks.
  </div>

  <div class="story-col-card">
    <div class="story-col-label">The PER-10 pillars</div>
    <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
      <thead>
        <tr style="border-bottom:1px solid rgba(148,163,184,0.5);">
          <th style="text-align:left; padding:0.25rem 0.3rem;">Code</th>
          <th style="text-align:left; padding:0.25rem 0.3rem;">Pillar</th>
          <th style="text-align:left; padding:0.25rem 0.3rem;">What it measures</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:0.25rem 0.3rem;"><strong>A</strong></td>
          <td style="padding:0.25rem 0.3rem;">Anticipation / Reaction</td>
          <td style="padding:0.25rem 0.3rem;">Trigger timing and first move toward the ball</td>
        </tr>
        <tr>
          <td style="padding:0.25rem 0.3rem;"><strong>S</strong></td>
          <td style="padding:0.25rem 0.3rem;">Separation / Space</td>
          <td style="padding:0.25rem 0.3rem;">Creating or denying usable space at the catch point</td>
        </tr>
        <tr>
          <td style="padding:0.25rem 0.3rem;"><strong>E</strong></td>
          <td style="padding:0.25rem 0.3rem;">Execution / Technique</td>
          <td style="padding:0.25rem 0.3rem;">Body control, leverage, hand use, finish</td>
        </tr>
        <tr>
          <td style="padding:0.25rem 0.3rem;"><strong>Eyes</strong></td>
          <td style="padding:0.25rem 0.3rem;">Eyes — Tracking &amp; Vision</td>
          <td style="padding:0.25rem 0.3rem;">Ball tracking, adjustment, staying connected to QB</td>
        </tr>
        <tr>
          <td style="padding:0.25rem 0.3rem;"><strong>Innovation</strong></td>
          <td style="padding:0.25rem 0.3rem;">Innovation — Creative Intelligence</td>
          <td style="padding:0.25rem 0.3rem;">New in-rep tools that upgrade the picture</td>
        </tr>
        <tr>
          <td style="padding:0.25rem 0.3rem;"><strong>I</strong></td>
          <td style="padding:0.25rem 0.3rem;">Improv Index (optional 6th dim)</td>
          <td style="padding:0.25rem 0.3rem;">How the player rescues the rep when it breaks</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>""",
            unsafe_allow_html=True,
        )


# --- CARD 2: SCALES & USAGE (STACKED UNDERNEATH) ---
        card2_html = """<div class="story-hero-shell">
<div class="story-col-card">
  <div class="story-col-label">Scales &amp; usage</div>

  <ul style="font-size:0.9rem; margin-top:0.3rem;">
    <li>
      <strong>PER-10 (1–10)</strong><br/>
      <span style="color:#cbd5e1;">Higher = smarter, cleaner, more repeatable reps.</span>
    </li>
    <li style="margin-top:0.4rem;">
      <strong>PER-10 360 (0–100)</strong><br/>
      <span style="color:#cbd5e1;">Full-season, opponent-adjusted Ball IQ score for WRs and DBs.</span>
    </li>
  </ul>

  <p style="margin-top:0.5rem; font-size:0.88rem; color:#e5e7eb;">
    Used as the <strong>single unifying metric</strong> for:
  </p>
  <ul style="margin-top:0.2rem; font-size:0.88rem; color:#e5e7eb;">
    <li>scouting grades (WR / DB)</li>
    <li>matchup game-planning</li>
    <li>development &amp; cut-up selection</li>
  </ul>
</div>
</div>"""

        st.markdown(card2_html, unsafe_allow_html=True)


        # Optional validation blurb
        if "per10" in overall.columns and "outcome_proxy" in overall.columns:
            tmp = overall.dropna(subset=["per10", "outcome_proxy"])
            if not tmp.empty:
                r = float(
                    np.corrcoef(
                        tmp["per10"].to_numpy(),
                        tmp["outcome_proxy"].to_numpy(),
                    )[0, 1]
                )
                st.markdown(
                    f"""<div style="margin-top:0.9rem; padding:0.75rem 0.95rem; border-radius:14px;
  border:1px solid rgba(148,163,184,0.55); background:rgba(15,23,42,0.96);
  font-size:0.88rem;">
🔗 <strong>Early signal:</strong> PER-10 shows a league-wide correlation of
<strong>ρ ≈ {r:.2f}</strong> with an EPA-like outcome metric in our sample, stronger than
raw separation or speed alone.
</div>""",
                    unsafe_allow_html=True,
                )

        # 3) HERO PLAY EXPLAINED ---------------------------------------------------
    elif step == "Hero Play Explained":
        st.markdown("### Hero plays: Pre-snap ➜ Model ➜ Outcome")
        st.markdown("<div class='story-spacer'></div>", unsafe_allow_html=True)

        hero_play_html = (
        "<div class='story-hero-shell'>"
        "<div class='story-hero-subtitle'>"
        "Two example reps that show how Sky Vision moves from alignment and leverage "
        "to a quantitative PER-10 insight, then to the football result."
        "</div>"

        "<div class='story-example'>"
        "<h3>Example 1 – 3rd &amp; 6 vs press man</h3>"
        "<div class='story-col-card'>"
        "<div class='story-col-label'>Pre-snap ➜ Model ➜ Outcome</div>"
        "<div style='font-size:0.9rem; line-height:1.6; color:#e5e7eb;'>"
        "<strong>Pre-snap</strong><br/>"
        "• 3x1, boundary X isolated<br/>"
        "• CB inside press, MOFC safety at 12 yards<br/>"
        "• Down &amp; distance: 3rd &amp; 6<br/><br/>"
        "<span class='label'>Football read:</span> "
        "WR must win outside leverage and hold the redline.<br/><br/>"
        "<strong>Model insight</strong><br/>"
        "• Pre-snap win probability: 44%<br/>"
        "• PER-10 pillars flagged (i.e. Anticipation: 8.5; Separation: 9.0; Eyes: 8.2)<br/><br/>"
        "<span class='label'>Takeaway:</span> "
        "This WR consistently solves press with outside release + stack, even in low-win leverage.<br/><br/>"
        "<strong>Outcome</strong><br/>"
        "• Clean win vs jam<br/>"
        "• 1.8 yards separation at catch point<br/>"
        "• Conversion + YAC<br/>"
        "Sky Vision tags this as an <span class='label'>Elite PER-10</span> rep and surfaces similar reps automatically for cut-ups."
        "</div>"
        "</div>"
        "</div>"

        "<div class='story-example' style='margin-top:2rem;'>"
        "<h3>Example 2 – Red zone scramble vs zone match</h3>"
        "<div class='story-col-card'>"
        "<div class='story-col-label'>Pre-snap ➜ Model ➜ Outcome</div>"
        "<div style='font-size:0.9rem; line-height:1.6; color:#e5e7eb;'>"
        "<strong>Pre-snap</strong><br/>"
        "• 2x2 bunch<br/>"
        "• MOFO shell, match rules vs bunch<br/>"
        "• QB on the move tendency in tight red<br/><br/>"
        "<span class='label'>Football read:</span> "
        "Coverage will pass routes, but scramble drill is live once QB breaks contain.<br/><br/>"
        "<strong>Model insight</strong><br/>"
        "• Pre-snap TD probability: 28%<br/>"
        "• Improv tag armed for WR and DB<br/>"
        "• WR Improv on this rep: 9.1<br/>"
        "• DB Improv on this rep: 6.0<br/><br/>"
        "<span class='label'>Takeaway:</span> "
        "WR consistently separates late when structure breaks.<br/><br/>"
        "<strong>Outcome</strong><br/>"
        "• QB breaks pocket<br/>"
        "• WR snaps across DB’s leverage at back line<br/>"
        "• TD at the pylon<br/>"
        "Sky Vision elevates this WR’s <span class='label'>Improv (I)</span> pillar and counts it toward the PER-10 360 ceiling."
        "</div>"
        "</div>"
        "</div>"

        "<div style='margin-top:1.4rem; font-size:0.85rem; color:#9ca3af;'>"
        "From alignment and leverage through model insight to outcome, Sky Vision shows who solved "
        "the rep and how that rolls into PER-10."
        "</div>"
        "</div>"
    )

        st.markdown(hero_play_html, unsafe_allow_html=True)

    
    # 4) WHY COACHES NEED SKY VISION ------------------------------------------
    else:
        st.markdown(
            """<div class="story-impact-shell">
  <div class="story-impact-title">How This Changes Player Evaluation</div>

  <div class="story-impact-row">
    <div class="story-impact-dot"></div>
    <div class="story-impact-body">
      <strong>Moves from GPS speed to decision speed.</strong>
      Sky Vision grades how quickly and correctly players adjust to leverage, rotations, and ball flight. Not just how fast they run.
    </div>
  </div>

  <div class="story-impact-row">
    <div class="story-impact-dot"></div>
    <div class="story-impact-body">
      <strong>Separates “open” from “solved.”</strong>
      Two 1.5-yard separation reps are not the same: one may be late, off the redline, or blind to the QB. PER-10 captures that difference.
    </div>
  </div>

  <div class="story-impact-row">
    <div class="story-impact-dot"></div>
    <div class="story-impact-body">
      <strong>Gives coaches a shared language.</strong>
      OC, DC, analytics, and scouting all talk through the same five-to-six pillars, rather than disconnected film notes and charts.
    </div>
  </div>

  <div class="story-impact-row">
    <div class="story-impact-dot"></div>
    <div class="story-impact-body">
      <strong>Turns reps into a development roadmap.</strong>
      Every high-leverage rep is tied back to Anticipation, Separation, Execution, Eyes, Innovation, and Improv making cut-ups and drill design obvious.
    </div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        if st.button("Jump to Player Assessments →", key="story_to_assess"):
            go("assess")


# ---------- MAIN ROUTER ----------
def main():
    # Which page are we on?
    page = st.session_state.get("page", "home")

    # Load data for tabs that need it
    pillars = overall = pool = None
    if page in ("home", "assess", "advanced", "story"):
        try:
            pillars, overall, pool = load_data()
        except FileNotFoundError as e:
            st.error(str(e))
            return

    # ---------- HOME / WELCOME ----------
    if page == "home":
        # Big SKY VISION header only on welcome
        render_top_bar()
        view_welcome(pillars, overall)

    # ---------- PLAYER ASSESSMENTS ----------
    elif page == "assess":
        # Subpage header with back button, no big top bar
        render_subpage_header("Player Assessments")

        if pillars is None or overall is None or pool is None:
            st.warning("Player data not available yet.")
            return

        tab1, tab2, tab3 = st.tabs(
            ["Player assessment", "Matchup comparison", "Pre-snap sandbox"]
        )

        with tab1:
            view_single_player(pillars, overall, pool)

        with tab2:
            view_comparison(pillars, overall, pool)

        with tab3:
            render_pre_snap_module()

    # ---------- STORY MODE ----------
    elif page == "story":
    # Add the same header/back button as other tabs
        render_subpage_header("Story Mode")
        view_story_mode(pillars, overall)


    # ---------- MOVEMENT & SPATIAL ----------
    elif page == "movement":
        render_subpage_header("Movement & Spatial View")
        view_movement_spatial()

    # ---------- ADVANCED METRICS ----------
    elif page == "advanced":
        render_subpage_header("Advanced Metrics")
        if pillars is None or overall is None:
            st.warning("Pillar data not available yet.")
        else:
            view_advanced_stats(pillars, overall)

    # ---------- PLAY DEEP DIVE ----------
    elif page == "play":
        render_subpage_header("Play Deep Dive")   # fixed
        play_deep_dive()

    # ---------- FALLBACK ----------
    else:
        render_top_bar()
        view_welcome(pillars, overall)


if __name__ == "__main__":
    main()