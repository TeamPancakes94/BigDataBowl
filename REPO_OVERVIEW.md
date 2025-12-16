# Repository Overview: AfterSnap IQ / BigDataBowl

This repository contains the codebase for **AfterSnap IQ**, an advanced analytics system designed for the NFL Big Data Bowl. The system evaluates Wide Receivers (WR) and Defensive Backs (DB) using a proprietary "PER-10" rating system based on player tracking data.

## 1. System High-Level Overview

The core purpose of this project is to transform raw NFL tracking data into actionable player insights using a two-stage process:

1.  **Pillar Extraction**: For every play, specific movement and reaction metrics are calculated to generate scores for 6 key pillars (Anticipation, Separation, Execution, Eyes, Innovation, Improv).
2.  **Bayesian Updating**: These play-level scores are fed into a Bayesian model to update the player's long-term "True Talent" rating, smoothing out noise and accounting for sample size.

The results are visualized in an interactive web application ("Sky Vision").

---

## 2. Directory Structure

- **`scripts/`**: The core data processing pipeline and application logic.
- **`notebooks/`**: Jupyter notebooks for research, prototyping, and analysis.
- **`docs/`**: Documentation defining the metrics, schema, and rules.
- **`data/`**: Raw input data (tracking data, game info).
- **`processed/`**: Intermediate and final output data (clean CSVs, scores).
- **`frames/`**: Generated images/frames from specific plays, used for visual verification.

---

## 3. Script Deep Dive & Data Pipeline

The data moves through the system in a linear pipeline. Here is exactly how the scripts work, step-by-step.

### Step 1: Data Merge (`scripts/01_merge_pipeline.py`)
**Goal**: Create a master roster and "core" file from dispersed weekly data.
1.  **Scan**: Finds all `input_2023_w*.csv` files in `train/`.
2.  **Concatenate**: Merges them into one DataFrame.
3.  **Validate**: Checks for required columns (`game_id`, `play_id`, `nfl_id`, `player_role`, etc.).
4.  **Extract Roster**: Filters down to unique player identities per play.
5.  **Output**: Writes `outputs/merged_core.csv`.

### Step 2: Metric Extraction (`scripts/02_extract_pillars.py`)
**Goal**: Calculate the 6 pillars for every player on every play.
1.  **Load Data**: Loads the raw tracking data (`input_2023_w*.csv`) and game event data (`output_2023_w*.csv`).
2.  **Standardize**: Renames columns to snake_case (e.g., `gameId` -> `game_id`) using `_standardize_cols`.
3.  **Ball Window Slicing**:
    *   Calls `find_throw_and_arrival` to identify the frame range where the ball is in width.
    *   Slices the tracking data to ONLY include frames between `pass_forward` and `pass_arrived` (or similar events). This is the "Ball-in-Air" window.
4.  **Pillar Calculation**:
    *   Calls helper functions in `per10.py` for each pillar (Anticipation, Separation, Execution, Eyes, Innovation, Improv).
    *   Passes the `window_df` (tracking data) to these functions.
5.  **Aggregation**:
    *   Merges all pillar scores into a single `per10_df`.
    *   Computes the composite `PER10` (5-pillar average) and `PER10_360` (6-pillar average).
6.  **Flagging**: Creates a `quick_instinct` binary flag if Eyes >= 8 AND (Innovation >= 8 OR Improv >= 7).
7.  **Output 1**: Writes `outputs/per10_traits.csv` (The final 1-10 scores).
8.  **Output 2**: Writes `outputs/merged_pillars.csv` (The raw underlying values like "milliseconds" or "yards" needed for the Bayesian step).

### Step 3: Bayesian Update (`scripts/03_update_posterior.py`)
**Goal**: Update "True Talent" ratings.
1.  **Input**: Reads `merged_pillars.csv`.
2.  **Model**: Uses a Beta-Binomial conjugate prior.
    *   **Prior**: Starts with a weak prior (e.g., average player).
    *   **Likelihood**: The new evidence from the current batch of plays.
    *   **Posterior**: The updated belief about the player's skill (0-100 scale).
3.  **Output**: Saves the updated posterior distributions (mean, variance) for each player.

---

## 4. Stat Pillar Calculation Breakdown

This section explains exactly **how** the raw Kaggle tracking data becomes a 1-10 score, based on the logic in `scripts/per10.py`.

### 1. Anticipation (A)
**Definition**: How quickly a player reacts to the ball being thrown.
*   **Input**: Player `x, y` over time, Ball landing `x, y`, Ball release frame.
*   **Step 1 (Vector Math)**: Calculate the vector from Player to Ball Landing Point (`bx`, `by`) and the Player's Movement vector (`dx`, `dy`).
*   **Step 2 (Alignment)**: Calculate dot product `alignment = dx*bx + dy*by`. High value means moving directly to the ball.
*   **Step 3 (Normalization)**: Normalize alignment to 0-1 range.
*   **Step 4 (Reaction Frame)**: Find the *first frame* where `alignment_norm > 0.25`. This is the "Reaction Frame".
*   **Step 5 (Delay)**: `frame_delay = Reaction Frame - Ball Release Frame`.
*   **Scoring**:
    *   <= 1 frame delay: **10**
    *   <= 3 frames: **8**
    *   <= 5 frames: **6**
    *   <= 7 frames: **4**
    *   > 7 frames: **2**

### 2. Separation (S)
**Definition**: Creating (WR) or closing (DB) space.
*   **Input**: Player `x, y`, Opponent `x, y` for each frame.
*   **Step 1 (Distance)**: Calculate Euclidean distance to the nearest opponent for every frame in the window.
*   **Step 2 (Delta)**: `Delta = Distance_End - Distance_Start`.
*   **Scoring (WR/Offense)**: Positive Delta (getting open) is good.
    *   >= 3 yds gained: **10**
    *   >= 2 yds: **8**
    *   >= 1 yd: **6**
*   **Scoring (DB/Defense)**: Negative Delta (closing gap) is good.
    *   <= -3 yds closed: **10**
    *   <= -2 yds: **8**
    *   <= -1 yd: **6**

### 3. Execution (E)
**Definition**: Movement efficiency and body control.
*   **Input**: Player `dir` (movement angle), `o` (orientation), `x, y`.
*   **Step 1 (Jerk/Smoothness - E1)**: Compute change in `dir` between frames. High change = high jerk. `E1 = 1 / (1 + mean_jerk)`.
*   **Step 2 (Body Alignment - E2)**: Compute difference between `dir` (where they are going) and `o` (where they are looking/facing). `E2 = 1 - (mean_misalignment / 180)`.
*   **Step 3 (Path Stability - E3)**: Compute dot product of consecutive movement vectors to see if the path is a straight line or wobbly. `E3 = 1 - (mean_wobble / 180)`.
*   **Scoring**: `Score = Average(E1, E2, E3) * 10`.

### 4. Eyes / Vision
**Definition**: Keeping eyes on the ball.
*   **Input**: Player `x, y`, Player `o`, Ball Landing `x, y`.
*   **Step 1 (Ball Angle)**: Calculate angle from Player to Ball Landing Spot.
*   **Step 2 (Gaze Error)**: `Error = |Player_Orientation - Ball_Angle|`.
*   **Step 3 (Components)**:
    *   **Angle**: How small is the average error?
    *   **Stability**: Standard deviation of the error (is the head shaking?).
    *   **Reaction**: How many frames until error < 20 degrees?
*   **Scoring**: `0.5 * Angle + 0.3 * Stability + 0.2 * Reaction`.

### 5. Innovation
**Definition**: Creative routing and non-linear movement.
*   **Input**: Player `x, y` path.
*   **Step 1 (Cuts)**: Identify sharp changes in direction (`mean_cut_angle`).
*   **Step 2 (Deviation)**: Fit a linear regression line to the path. Calculate `deviation` (RMSE) of actual path vs line. Measures "non-linearity".
*   **Step 3 (Effectiveness)**: Did this non-linear path actually improve position relative to the ball?
*   **Scoring**: `0.4 * Cuts + 0.3 * Deviation + 0.3 * Effectiveness` (Scaled 0-10).

### 6. Improv (I)
**Definition**: Recovering from broken plays or chaos.
*   **Input**: Player Speed `s`, Direction `dir` changes.
*   **Step 1 (Disruption - D)**: Measure magnitude of sudden speed changes (`diff(s)`) and direction flips (`diff(dir)`).
*   **Step 2 (Recovery - R)**: If a disruption occurs, how many frames until movement stabilizes (direction change < 10 deg)? Faster recovery = higher score.
*   **Step 3 (Gain - G)**: Position improvement after the disruption.
*   **Scoring**: `0.4 * D + 0.3 * R + 0.3 * G` (Scaled 0-10).

---

## 5. From Kaggle Data to Stats (Data Lineage)

1.  **Raw Input**: Kaggle `tracking_week_*.csv` provides 10Hz data: `(gameId, playId, nflId, frameId, x, y, s, a, dis, o, dir)`.
2.  **Event Anchors**: We find the frames where `event == 'pass_forward'` and `event == 'pass_arrived'`.
3.  **Filtration**: We drop all frames outside this window. We now have "Ball-in-Air" tracking data.
4.  **Transformation**:
    *   **Positional**: $(x, y)$ coordinates are used to derive distance and separation vectors.
    *   **Angular**: $(dir, o)$ are used to derive Eyes (looking at ball) and Execution (body control).
    *   **Temporal**: Frame counts are converted to milliseconds (1 frame = 100ms) for Reaction Time (Anticipation).
5.  **Normalization**: Raw metrics (e.g., "Wait time = 200ms") are mapped to 1-10 scores using the thresholds defined in `per10.py` (e.g., <=100ms = 10/10).
6.  **Contextualization**: Scores are split by Role (Offense vs Defense) because "closing separation" is good for one but bad for the other.

This pipeline ensures that every 1-10 stat is directly traceable back to the physics of the player's movement on the field.
