# Improv Index (I) — Calculation Methodology

The **Improv Index (I)** quantifies a player’s ability to **adapt, recover, and improve outcome after a play breaks its original structure**. It is designed to capture **reactive athletic intelligence** using player tracking data.

Each player's Improv Index is computed **per play** and scaled to a **0–10 range**.

---

## Overview of Components

The Improv Index is a weighted combination of three components:

- **D — Disruption Detection (40%)**
- **R — Recovery Speed (30%)**
- **G — Outcome Gain (30%)**

\[
\textbf{Improv} = 0.4D + 0.3R + 0.3G
\]

---

## 1. Disruption Detection (D)

Disruption measures how violently a player’s movement changes during the play. It captures both:

- **Sudden speed changes**
- **Sudden directional changes**

### Speed Disruption
The absolute frame-to-frame speed change is computed:

\[
\Delta s_t = |s_t - s_{t-1}|
\]

The mean speed disruption is:

\[
\overline{\Delta s} = \text{mean}(\Delta s)
\]

---

### Directional Disruption
Direction is computed from positional changes:

\[
\theta_t = \tan^{-1}\left(\frac{y_t - y_{t-1}}{x_t - x_{t-1}}\right)
\]

Angular change between frames:

\[
\Delta \theta_t = \left|(\theta_t - \theta_{t-1} + 180) \bmod 360 - 180\right|
\]

Mean directional disruption:

\[
\overline{\Delta \theta} = \text{mean}(\Delta \theta)
\]

---

### Raw Disruption Score

\[
D_{raw} = \overline{\Delta s} + \frac{1}{2}\overline{\Delta \theta}
\]

This is normalized to a **0–10 scale**:

\[
D = \text{clip}\left(\frac{D_{raw}}{5}, 0, 10\right)
\]

> The constant *5* represents a typical upper-bound observed in NFL tracking data.

---

## 2. Recovery Speed (R)

Recovery Speed measures how quickly a player stabilizes **after the largest movement disruption**.

### Step 1 — Identify the Chaos Moment
The frame of maximum directional change is identified:

\[
t_{disrupt} = \arg\max(\Delta \theta)
\]

### Step 2 — Measure Post-Disruption Stability
Only frames after the disruption are analyzed. Recovery time is how long it takes for the directional variance to drop below a stability threshold (10 degrees).

### Recovery Score

\[
R = \text{clip}(10 - 1.5 \times \text{recovery\_frames}, 0, 10)
\]

- Faster recovery → higher **R**
- Slower recovery → lower **R**

If insufficient post-disruption frames exist, **R is assigned a neutral value of 5**.

---

## 3. Outcome Gain (G)

Outcome Gain measures whether the player’s improvisation **actually improved the play outcome**.

### Offensive Players

For offensive players, outcome gain is measured using **distance to the ball landing point**:

\[
G = \text{clip}(d_{start} - d_{end}, 0, 10)
\]

Where:
- \( d_{start} \) = initial distance to the landing point  
- \( d_{end} \) = final distance to the landing point  

Positive values indicate **improvement toward the target**.

---

### Defensive Players

For defenders, outcome gain is measured using **separation reduction**:

\[
G = \text{clip}(s_{start} - s_{end}, 0, 10)
\]

Where:
- \( s_{start} \) = initial target separation  
- \( s_{end} \) = final target separation  

Positive values indicate **successful closing on the receiver**.

---

## Final Improv Index Formula

\[
\boxed{
\textbf{Improv} = 0.4D + 0.3R + 0.3G
}
\]

Each player receives a **single Improv score per play**, reflecting:

- How chaotic their movement became,
- How fast they regained control,
- And whether their improvisation improved the outcome.

---

## Dataset Output

The final output table contains:

| Column | Description |
|--------|-------------|
| `play_id` | Unique play identifier |
| `nfl_id` | Unique player identifier |
| `improv_score` | Final Improv Index (0–10) |

---

## Interpretation Guide

| Improv Score | Meaning |
|--------------|---------|
| 0–3 | Little or no effective adaptation |
| 4–6 | Moderate improvisation |
| 7–8 | Strong adaptive response |
| 9–10 | Elite improvisational performance |

---

This metric is intended to quantify **in-the-moment creativity, recovery, and reactive decision-making** using pure movement physics from tracking data.
