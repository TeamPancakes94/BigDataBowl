import pandas as pd
import numpy as np
from pathlib import Path


POSITION_WEIGHTS = {
    "Offense": {
        "A": 0.25,
        "S": 0.30,
        "E": 0.15,
        "Eyes": 0.10,
        "Improv": 0.20,
    },
    "Defense": {
        "A": 0.10,
        "S": 0.35,
        "E": 0.15,
        "Eyes": 0.25,
        "Improv": 0.15,
    },
}


def compute_weighted_trait_score(row):
    """
    row = one row of per-play trait data for one player.
    Uses whatever pillars are present (A, S, E, Eyes, Improv),
    skipping NaNs and re-normalizing the weights.
    """
    role = row["player_side"]
    weights = POSITION_WEIGHTS.get(role)

    if weights is None:
        return np.nan

    numer = 0.0
    denom = 0.0

    for pillar, w in weights.items():
        val = row.get(pillar)
        if pd.notna(val):
            numer += w * float(val)
            denom += w

    if denom == 0:
        # no usable pillars for this play
        return np.nan

    score = numer / denom
    return float(round(score, 3))



def bayesian_update(prior_mean, prior_var, observation, obs_var=1.0):
    """
    prior_mean: current belief of player's true ability
    prior_var: uncertainty in that belief
    observation: new play weighted score
    obs_var: noise in a single play (default = 1)
    """
    posterior_mean = (
        (obs_var * prior_mean + prior_var * observation) /
        (prior_var + obs_var)
    )
    posterior_var = (prior_var * obs_var) / (prior_var + obs_var)
    return posterior_mean, posterior_var


def run_bayesian_player_updates(per_play_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns required:
        ['nfl_id', 'game_id', 'play_id', 'weighted_trait']

    Output:
        DataFrame with columns:
        ['nfl_id', 'game_id', 'play_id', 'bayesian_rating', 'bayesian_uncertainty']
    """
    per_play_df = per_play_df.sort_values(["nfl_id", "game_id", "play_id"])

    player_state = {}
    results = []

    for _, row in per_play_df.iterrows():
        pid = row["nfl_id"]
        obs = row["weighted_trait"]

        # If the weighted trait is NaN, skip updating on this play
        if pd.isna(obs):
            continue

        # Initialize prior
        if pid not in player_state:
            player_state[pid] = {
                "mean": 5.0,   # neutral prior
                "var": 4.0,    # high uncertainty
            }

        prior = player_state[pid]
        new_mean, new_var = bayesian_update(
            prior["mean"], prior["var"], obs
        )

        player_state[pid]["mean"] = new_mean
        player_state[pid]["var"] = new_var

        results.append({
            "nfl_id": pid,
            "game_id": row["game_id"],
            "play_id": row["play_id"],
            "bayesian_rating": round(new_mean, 3),
            "bayesian_uncertainty": round(new_var, 3),
        })

    # even if results is empty, still define the columns
    return pd.DataFrame(
        results,
        columns=["nfl_id", "game_id", "play_id",
                 "bayesian_rating", "bayesian_uncertainty"],
    )


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    OUT_DIR = ROOT / "outputs"

    # 1) Load per-play traits from the canonical file
    traits_path = OUT_DIR / "per10_traits.csv"
    traits = pd.read_csv(traits_path)

    # Normalize / check columns
    traits.columns = [c.strip() for c in traits.columns]

    required = [
        "play_id", "nfl_id", "game_id", "player_side",
        "A", "S", "E", "Eyes", "Improv",
    ]
    missing = [c for c in required if c not in traits.columns]
    if missing:
        raise KeyError(f"per10_traits.csv missing columns: {missing}")

    # 2) Compute weighted_trait per play using POSITION_WEIGHTS
    traits["weighted_trait"] = traits.apply(compute_weighted_trait_score, axis=1)

    # 3) Run Bayesian updates over time
    per_play_for_bayes = traits[["nfl_id", "game_id", "play_id", "weighted_trait"]].copy()
    bayes_by_play = run_bayesian_player_updates(per_play_for_bayes)

    # 4) Take the last rating per player as the final rating
    bayes_final = (
        bayes_by_play
        .sort_values(["nfl_id", "game_id", "play_id"])
        .groupby("nfl_id", as_index=False)
        .tail(1)
    )

    # Rename for consistency with other files
    bayes_final = bayes_final.rename(columns={"nfl_id": "player_id"})

    # 5) Scale bayesian_rating to 0–100 for easier display
    bayes_final["overall_0_100"] = bayes_final["bayesian_rating"] * 10.0

    # 6) Save to outputs
    out_path = OUT_DIR / "bayesian_player_ratings.csv"
    bayes_final.to_csv(out_path, index=False)
    print("Saved Bayesian player ratings →", out_path.resolve())
