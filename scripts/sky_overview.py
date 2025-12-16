# sky_overview.py

from __future__ import annotations

import textwrap
import streamlit as st
import streamlit.components.v1 as components


# Pillar definitions ------------------------------------

PILLARS = [
    {
        "code": "A",
        "name": "Anticipation / Reaction",
        "short": "Who moves first toward the ball after release. Frame-to-frame timing and intent.",
        "long_body": """
**Core question**

Who moves first toward the ball once it leaves the quarterback’s hand?

We track frame-to-frame timing: does the player trigger early, beat their
matchup to the catch path, and consistently fire in the right direction?
High A scores flag players who see the throw coming, react first, and turn
that first move into an advantage at the arrival point.

""",
        "is_core": True,
    },
    {
        "code": "S",
        "name": "Separation / Space",
        "short": "Who creates or denies usable space at the catch point.",
        "long_body": """
**Core question**

Who creates (WR) or denies (DB) functional distance at the catch point?

We measure whether the matchup arrives with enough separation to finish the
play cleanly, or whether the defender closes windows and compresses space.
High S scores go to players who consistently create throwing alleys or erase
them before the ball arrives.

""",
        "is_core": True,
    },
    {
        "code": "E",
        "name": "Execution / Technique",
        "short": "How optimal body control, leverage, and mechanics are under pressure.",
        "long_body": """
**Core question**

How cleanly does the player apply technique, leverage, and mechanics under pressure?

Execution captures body control, leverage, hand usage, hip fluidity, and how well
the player stays true to the assignment. High E scores highlight players who
win with repeatable technique rather than one-off chaos.

""",
        "is_core": True,
    },
    {
        "code": "Eyes",
        "name": "Eyes — Tracking & Vision",
        "short": "Who identifies and tracks the ball earliest and most continuously.",
        "long_body": """
**Core question**

Who locates and tracks the ball earliest and most accurately?

Eyes measures how early and cleanly a player identifies the ball in flight,
stays locked on it, and adjusts their path accordingly. Players who see the
ball first win earlier—they position their body, hands, and leverage before
their matchup can react.

""",
        "is_core": True,
    },
    {
        "code": "Innovation",
        "name": "Innovation — Creative Intelligence",
        "short": "Did the player generate a new in-rep solution that improved the outcome?",
        "long_body": """
**Core question**

Did the player invent a new in-rep solution that meaningfully improved outcome probability?

Innovation captures intentional creativity: micro-stems, tempo changes,
leverage fakes, hip flips, and tools that alter the picture without
breaking structure. High Innovation scores flag players who proactively
upgrade the rep.

""",
        "is_core": True,
    },
    {
        "code": "I",
        "name": "Improv Index (I)",
        "short": "Reactive adaptability once the original plan breaks. How well the player rescues the rep.",
        "long_body": """
**What Improv Index measures**

On top of the core five pillars, Sky Vision tracks the Improv Index (I)—a
separate score for how effectively a player rescues the rep once the
original plan breaks.

Improv captures reactive adaptability: instinctive choices made in split
seconds when structure collapses. We look for players who:
- React quickly to unexpected movement or pressure,
- Intelligently alter route path, leverage, or pacing mid-rep,
- Maintain control while creating a new answer on the fly.

Improv Index (I) stands alone as its own score, but can also be blended into
a PER-10 360 style view when you want all six dimensions.

*Note: Improv is **not** one of the core five PER-10 pillars; it is an
optional sixth dimension layered on top.*
""",
        "is_core": False,
    },
]

def render_sky_pipeline():
    pipeline_html = """
    <style>
      .sky-pipeline-shell {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        border-radius: 26px;
        padding: 20px;
        background: radial-gradient(circle at top left, #1e293b, #020617);
        border: 1px solid rgba(148,163,184,0.45);
        color: #e5e7eb;
      }
      .sky-pipeline-inner {
        border-radius: 22px;
        padding: 20px 24px 24px;
        background: rgba(15,23,42,0.96);
        border: 1px solid rgba(30,64,175,0.7);
      }
      .sky-pipeline-kicker {
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 1.1rem;
      }
      .sky-pipeline-row {
        display: flex;
        gap: 1.75rem;
        align-items: stretch;
      }
      .sky-pipeline-left {
        flex: 0 0 32%;
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.55);
        font-size: 0.9rem;
      }
      .sky-pipeline-left-label {
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #bfdbfe;
        margin-bottom: 0.45rem;
      }
      .sky-pipeline-left-copy {
        color: #e5e7eb;
      }
      .sky-pipeline-right {
        flex: 1;
        display: flex;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
      }
      .sky-pipeline-step {
        flex: 1 1 28%;
        min-width: 180px;
        padding: 12px 14px;
        border-radius: 16px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.55);
        font-size: 0.86rem;
      }
      .sky-pipeline-step-label {
        font-size: 0.75rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #bfdbfe;
        margin-bottom: 0.35rem;
      }
      .sky-pipeline-step-body {
        color: #e5e7eb;
        line-height: 1.55;
      }
      .sky-pipeline-arrow {
        font-size: 1.4rem;
        padding: 0 6px;
        color: #9ca3af;
      }
      .sky-pipeline-outcomes {
        margin-top: 1.4rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
      }
      .sky-pipeline-pill {
        font-size: 0.8rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.7);
        background: rgba(15,23,42,0.9);
        color: #e5e7eb;
        white-space: nowrap;
      }
      @media (max-width: 900px) {
        .sky-pipeline-row {
          flex-direction: column;
        }
        .sky-pipeline-left {
          flex: 1 1 auto;
        }
        .sky-pipeline-right {
          flex-direction: column;
        }
      }
    </style>

    <div class="sky-pipeline-shell">
      <div class="sky-pipeline-inner">
        <div class="sky-pipeline-kicker">
          FROM RAW TRACKING DATA TO LIVE FOOTBALL INTELLIGENCE
        </div>

        <div class="sky-pipeline-row">

          <div class="sky-pipeline-left">
            <div class="sky-pipeline-left-label">Tracking data</div>
            <div class="sky-pipeline-left-copy">
              Raw player &amp; ball coordinates for every frame of every play including
              alignments, motions, coverage shells, and route paths.
            </div>
          </div>

          <div class="sky-pipeline-right">

            <div class="sky-pipeline-step">
              <div class="sky-pipeline-step-label">Feature engineering</div>
              <div class="sky-pipeline-step-body">
                We reconstruct routes and coverages, then derive leverage, separation,
                timing, and ball-flight features at the rep level.
              </div>
            </div>

            <div class="sky-pipeline-arrow">➜</div>

            <div class="sky-pipeline-step">
              <div class="sky-pipeline-step-label">Pillar scores</div>
              <div class="sky-pipeline-step-body">
                Each targeted rep is graded on Sky Vision’s football-native pillars:
                Anticipation, Separation, Execution, Eyes, Innovation, and Improv.
              </div>
            </div>

            <div class="sky-pipeline-arrow">➜</div>

            <div class="sky-pipeline-step">
              <div class="sky-pipeline-step-label">PER-10 model</div>
              <div class="sky-pipeline-step-body">
                Rep-level pillars roll into stable PER-10 360 player ratings
                for every WR and DB in the league.
              </div>
            </div>

          </div>
        </div>

        <div class="sky-pipeline-outcomes">
          <div class="sky-pipeline-pill">PER-10 pillars per rep</div>
          <div class="sky-pipeline-pill">PER-10 360 player ratings</div>
          <div class="sky-pipeline-pill">Analyst dashboard · Deep scouting</div>
          <div class="sky-pipeline-pill">Broadcast dashboard · Live overlays</div>
        </div>
      </div>
    </div>
    """

    components.html(pipeline_html, height=420, scrolling=False)


# Main welcome page -------------------
def view_welcome(pillars, overall):
    """
    Landing / welcome page with hero, project overview, pipeline,
    pillars grid, and PER-10 summary.

    `pillars` and `overall` are accepted for API compatibility but not required here.
    """

    # ---------- HERO ----------
    st.markdown(
        textwrap.dedent(
            """
            <div class="sky-hero-wrapper">
              <div class="sky-hero-kicker">
                TRACKING THE BALL IN THE AIR. QUANTIFYING LEVERAGE ON THE GROUND.
              </div>

              <div class="sky-hero-title">
                SKY VISION: REDEFINING PLAYER EVALUATION
              </div>

              <p class="sky-hero-body">
                Sky Vision transforms complex tracking, spatial, and movement data into clear,
                actionable intelligence. Instead of relying only on box scores or isolated clips,
                we quantify how players create and deny separation, execute assignments, and
                finish plays across every targeted rep in the season.
              </p>

              <div class="sky-hero-chips">
                <span class="sky-hero-chip">Route-level evaluation</span>
                <span class="sky-hero-chip">WR &amp; DB matchups</span>
                <span class="sky-hero-chip">PER-10 360 efficiency</span>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # ---------- PROJECT OVERVIEW ----------
    st.markdown("### Project overview")
    st.write(
        "The system ingests tracking and event data for every play, reconstructs routes and coverages, "
        "and then scores each rep across football-native pillars. Those rep-level traits roll up into "
        "stable player evaluations so that we can answer questions like: **Who consistently wins leverage? "
        "Who finishes at the catch point? Where does separation truly come from?**"
    )

    # ---------- SKY VISION PIPELINE ----------
    st.markdown("### Sky Vision pipeline")
    render_sky_pipeline()

    # first glowing divider
    st.markdown(
        '<hr class="sky-section-divider sky-top-divider">',
        unsafe_allow_html=True,
    )

    # ---------- PILLAR GRID ----------
    st.markdown("### The Sky Vision pillars")
    st.markdown("_A unified framework for receivers and defensive backs._")

    rows = [PILLARS[:3], PILLARS[3:]]

    if "active_pillar_code" not in st.session_state:
        st.session_state["active_pillar_code"] = None

    st.markdown('<div class="pillar-grid">', unsafe_allow_html=True)

    for row in rows:
        cols = st.columns(3, gap="large")
        for col, pillar in zip(cols, row):
            with col:
                label = f"{pillar['name']}\n\n{pillar['short']}"
                clicked = st.button(label, key=f"pill_{pillar['code']}")
                if clicked:
                    st.session_state["active_pillar_code"] = pillar["code"]

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- IMPROV BLURB ----------
    st.markdown("#### Improv Index (I): Optional 6th Dimension")
    st.write(
        "On top of the core five pillars, Sky Vision tracks the **Improv Index (I)**. This tracks how effectively "
        "a player rescues the rep when structure collapses. Improv stands alone as a reactive adaptability score, "
        "but can be blended into PER-10 360 when you want the full six-dimension view."
    )

    # second glowing divider
    st.markdown('<hr class="sky-section-divider">', unsafe_allow_html=True)

    # ---------- PER-10 SUMMARY ----------
    st.markdown("### How PER-10 scores work (at a glance)")
    st.markdown(
        """
- **PER-10** averages the five core pillars:  
  **Anticipation / Reaction (A)**, **Separation / Space (S)**,
  **Execution / Technique (E)**, **Eyes — Tracking & Vision**,
  and **Innovation — Creative Intelligence**.

- **Improv Index (I)** stands alone as a reactive adaptability score, but can be blended into a **PER-10 360** view when you want a full
  six-dimension intelligence profile.
        """
    )

    # ---------- PILLAR DETAIL POP-UP ----------
    active_code = st.session_state.get("active_pillar_code")
    if active_code:
        pillar = next(p for p in PILLARS if p["code"] == active_code)

        @st.dialog(pillar["name"])
        def _show_pillar_dialog():
            st.write(pillar["long_body"])
            if st.button("Close"):
                st.session_state["active_pillar_code"] = None

        _show_pillar_dialog()

    # ---------- TECHNICAL OVERVIEW ----------
    with st.expander(" How Sky Vision Works (Technical Overview)"):
        st.markdown(
            """
**1. Route & coverage reconstruction**
- Use tracking data (x,y, speed, orientation) to rebuild route stems and DB leverage.
- Compute redline, spacing, and alignment context per frame.

**2. Feature engineering**
- Separation curve (frame-by-frame)
- Leverage delta (inside/outside wins)
- Timing markers (break step, trigger, first move toward ball)
- Ball-flight tracking (eyes, adjustment angles)

**3. PER-10 pillar scoring**
Each rep is labeled on:
- **A** Anticipation
- **S** Separation
- **E** Execution
- **Eyes** Tracking
- **Innovation** (creative upgrades)
- **I** Improv (structure breaks)

**4. PER-10 360**
- Bayesian aggregation across reps
- Opponent-adjusted, game-context–adjusted IQ measure

**5. Validation**
- CORR(PER-10, outcome_proxy) ≈ strong positive
- Stronger than raw separation or speed
            """
        )
