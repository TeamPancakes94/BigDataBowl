<h1> Section 1 </h1>

<h2>Motivation, Problem, and Approach </h2>

<h3>Motivation:</h3>

NFL and NCAA evaluations still rely heavily on subjective notes such as “good ball skills,” “tight coverage,” or “quick reaction.” These descriptions lack quantitative detail about why a receiver wins a rep or how a defensive back stays connected at the catch point.

Traditional metrics capture physical outcomes (speed, alignment, separation), but not the underlying technical and cognitive layers that actually determine ball success. Sky Vision is designed to translate these hidden factors into measurable, repeatable data.

<h3>Problem:</h3>

Scouting and coaching face three core gaps:

1. No standardized measurement of ball-specific skill

Terms like “tracking,” “instincts,” and “reaction” are vague and inconsistent across evaluators.

2. Outcome-based grading hides the process

A catch or pass breakup doesn’t capture the timing, technique, anticipation, or decision-making behind it.

3. Creativity and adaptation are unmeasured

Micro-adjustments, deception, tempo changes, and recovery behaviors significantly influence ball outcomes but are not quantified by current analytics.

These gaps make it difficult to:

- compare WRs and DBs across reps,
- identify instinctive vs. mechanical players,
- model ball outcome probability with technical fidelity.

<h3>Approach</h3>

Sky Vision solves these problems using a five-pillar Ball IQ model (PER-10) and a separate Improv Index to score every targeted rep with both technical and cognitive depth.

**Five Pillars of Ball IQ**

Each category is scored from 1–10:

- A — Anticipation / Reaction: Who moves first toward the ball after release?
- S — Separation / Space: Who creates or denies functional distance at the catch point?
- E — Execution / Technique: How optimal are body control, leverage, and hand usage?
- Eyes — Tracking & Vision: Who identifies and tracks the ball earliest and most accurately?
- Innovation — Creative Intelligence: Did the player generate a new in-rep solution that improved outcome probability?

**PER-10 = ROUND((A + S + E + Eyes + Innovation) / 5)**

**Improv Index (I)**

- A standalone score capturing reactive adaptability when the rep leaves script. Used optionally in an integrated metric:

**PER-10 360 = ROUND((A + S + E + Eyes + Innovation + I) / 6)**

**Frame-Level Workflow**

- Record rep (1s pre-snap → 1s post-catch).
- Tag critical frames (Snap, Release, Arrival, Improv Moment).
- Input scores (A, S, E, Eyes, Innovation, I).
- Auto-calculate PER-10 and PER-10 360.
- Export dataset to CSV; overlay Sky Vision film.
- Quick Instinct badge triggers when criteria are met.

**Standardized Output**

- Each play becomes structured data with transparent scoring tied directly to film.
- This enables consistent comparison, clear scouting language, and robust predictive modeling.

**Summary**

Sky Vision converts qualitative film traits into a quantitative language of ball intelligence.
By isolating anticipation, space creation, technique, vision, creativity, and adaptive response, the system reveals how a rep was won, not just what happened.

The result is a transparent, scalable framework for scouting, coaching, and analytics that bridges film and data with precision.

<h1> Section 3 <h1>

<h2>PER-10, Improv, and Bayesian System — Explanation</h2>

### PER-10 Overview

The PER-10 model is a composite scoring system that evaluates a player’s Ball IQ during each play.

PER-10 is a simple, interpretable 1–10 score based on six pillars:

- A — Anticipation
- S — Separation (WR creation / DB denial)
- E — Execution + Technique
- Eyes — Ball Tracking / Vision
- Innovation — Creative, effective mid-play adjustments
- Improv — Disruption response & off-script performance

Each pillar already produces a 1–10 value.
PER-10 is simply the mean of the six pillars, rounded to the nearest whole number

### How Each PER-10 Pillar Is Calculated

**Anticipation (A)**

**What it measures:**
How quickly the player begins moving toward the ball after the ball is released.

**Inputs**:
- Frame of ball release
- First frame of purposeful movement toward ball
- Change in directional velocity
- Time difference between WR and DB reactions

**Calculation:**

- Compute Δt = time between ball release and player's first directional movement toward catch point.
- Convert to a percentile within the sample of the rep (WR vs DB).
- Map percentile → 1–10 scale.

**Interpretation:**

- 10 = instant movement, earliest reaction
- 1 = worst reaction or no identifiable reaction

**Separation / Space (S)**

**What it measures:**
Whether the player created (WR) or denied (DB) usable space at the catch point.

**Inputs:**

- Player positions (x, y) for each frame
- Horizontal/vertical separation at catch point
- Relative speed & acceleration before arrival

**Calculation:**

Track player-to-opponent distance across frames.

- For WR: more positive separation → higher score.
- For DB: less separation or separation reduction → higher score.

Normalize distance vs expected baseline → convert to 1–10 score.

**Interpretation:**

- 10 = WR creates clear window; DB closes distance completely
- 1 = WR smothered; DB burned

**E — Execution**

**What It Measures**

Overall technical quality of movement, including efficiency, body control, and ability to maintain leverage through the rep.

**Inputs**

- Direction smoothness (how jerky or clean movement is)
- Orientation control (how well body alignment matches movement)
- Path consistency (how stable footwork and angles are throughout the rep)
- How the Score Is Calculated
- Combine three elements:
- Smoothness of directional changes
- Body alignment consistency
- Stability of path / footwork
- More stable, controlled, and efficient movement → higher Execution score.

**Interpretation**

- 10: Textbook mechanics; fully controlled movement
- 7–9: Strong technique with minor inefficiencies
- 4–6: Occasional balance or leverage issues
- 1–3: Poor control, unstable steps, inefficient path

**Eyes — Ball Tracking / Vision**

**What It Measures**

How early and how accurately a player locates, tracks, and orients to the ball as it travels.

**Inputs**

- Angle between player head orientation and ball trajectory
- Stability of ball tracking
- Time until ball is visually picked up
- How the Score Is Calculated
- Based on three weighted components:
- Alignment – How directly the player is looking at the ball
- Tracking Stability – How steady their tracking is across frames
- Reaction Time – How quickly they locate the ball after release
- Higher alignment + earlier tracking + stable vision = higher score.

**Interpretation**

- 10: Instant, continuous ball tracking

- 7–9: Early pickup, good discipline

- 4–6: Late or inconsistent tracking

- 1–3: Never properly locates ball

**Innovation**

**What It Measures**

Creative, intentional adjustments that improve the outcome of the rep beyond the designed route or coverage.

This captures smart creativity, not randomness.

**Inputs**

- Size and timing of directional changes (cuts, pivots, micro-stems)
- Deviation from predicted / linear movement
- Impact of the adjustment (space created, angle improved, contest ability)

**How the Score Is Calculated**

Three components contribute:

- Creativity of movement (quality of cuts / changes)
- Non-linearity (use of deception or nonstandard pacing)
- Effectiveness (does the adjustment improve outcome?)
- High-impact, intentional innovations → high score.

**Interpretation**

- 10: Clear, decisive innovation that changes the play
- 8–9: Strong improvisational idea with real effect
- 6–7: Minor but useful creativity
- 4–5: Neutral; adds little
- 1–3: Counterproductive or confusing adjustment

**Improv (I)**

**What It Measures**

Real-time adaptability when the rep breaks from the script — reacting to unexpected movement, pressure, or ball placement.

Innovation = deliberate creativity
Improv = reactive creativity

**Inputs**

- Sudden adjustments after unplanned events
- Recovery steps, redirections, bailout paths
- How well the player salvages the rep
- How the Score Is Calculated
- Evaluate how often and how effectively the player makes successful reactive adjustments.
- Good improv means the player can recover efficiently under chaos.

**Interpretation**

- 10: Elite adaptability; turns broken plays into wins
- 7–9: Strong reaction ability; rarely panics
- 4–6: Sometimes recovers, sometimes struggles
- 1–3: Poor reaction; breakdowns cause negative outcomes

**Overall Use**

All six scores create the analytical backbone for **PER-10 and PER-10 360**:

**PER-10 = average(A, S, E, Eyes, Innovation)**

**PER-10 360 = average(A, S, E, Eyes, Innovation, Improv)**


Together they reflect:

- speed of recognition
- technical skill
- spatial performance
- visual intelligence
- creative problem solving
- chaos adaptability

## Bayesian System—  Trait Tracking Over Time

The Bayesian layer is the engine of the player rating system.

It answers one question:

“Based on every play we’ve seen, what is our best estimate of this player’s true talent?”

Instead of averaging or summing ratings, it uses Bayesian updating, which is a method for refining a belief every time you see new evidence.

**What the Bayesian Update Is**

A system that:

- Starts with a neutral assumption about a player (middle score, high uncertainty)
- Looks at each new play’s trait score
- Updates the estimated “true ability”
- Reduces uncertainty as more plays are observed

It acts like a coach gaining confidence:

- Early on, one good or bad play changes what you think a lot
- Later on, when you’ve seen many plays, one new play shifts your opinion only a little

**What the Bayesian Update Is For**

It allows the system to:

- Track player talent continuously
- Smooth out randomness from single plays
- Get more confident as more data comes in
- Handle players with few snaps and players with hundreds
- React appropriately to hot streaks or declining performance

In simple terms:

- Stability: prevents overreacting to one great or terrible play
- Adaptability: still adjusts as a player genuinely improves or worsens
- Fairness: players who play more get more accurate ratings
- Uncertainty tracking: you know how confident you are in the rating, not just the rating itself

**How Scores Are Interpreted**

Just like the trait scores:

- 10 = exceptional player-level ability
- 1 = poor ability
- 5 = league-average (starting point for everyone)
