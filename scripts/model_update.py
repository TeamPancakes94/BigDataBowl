import pandas as pd
import numpy as np

POSITION_WEIGHTS = {
    "Offense": {
        "A": 0.25,
        "S": 0.30,
        "E": 0.15,
        "Eyes": 0.10,
        "Improv": 0.20
    },
    "Defense": {
        "A": 0.10,
        "S": 0.35,
        "E": 0.15,
        "Eyes": 0.25,
        "Improv": 0.15
    }
}

def compute_weighted_trait_score(row):
    """
    row = one row of per-play trait data for one player
    Must include:
    ['player_side', 'A', 'S', 'E', 'Eyes', 'Improv']
    """
    role = row["player_side"]
    weights = POSITION_WEIGHTS.get(role)

    if weights is None:
        return np.nan

    score = (
        weights["A"]            * row["A"] +
        weights["S"]            * row["S"] +
        weights["E"]            * row["E"] +
        weights["Eyes"]         * row["Eyes"] +
        weights["Improv"]       * row["Improv"]
    )

    return round(float(score), 3)

def bayesian_update(prior_mean, prior_var, observation, obs_var=1.0):
    """
    prior_mean: current belief of player's true ability
    prior_var: uncertainty in that belief
    observation: new play weighted score
    obs_var: noise in a single play (default = 1)

    Returns:
    updated_mean, updated_variance
    """

    posterior_mean = (
        (obs_var * prior_mean + prior_var * observation) /
        (prior_var + obs_var)
    )

    posterior_var = (prior_var * obs_var) / (prior_var + obs_var)

    return posterior_mean, posterior_var

def run_bayesian_player_updates(per_play_df):
    """
    Input columns required:
    ['nfl_id', 'game_id', 'play_id', 'weighted_trait']

    Output:
    player-level Bayesian ratings
    """

    per_play_df = per_play_df.sort_values(["nfl_id", "game_id", "play_id"])

    player_state = {}
    results = []

    for _, row in per_play_df.iterrows():
        pid = row["nfl_id"]
        obs = row["weighted_trait"]

        if np.isnan(obs):
            continue

        # Initialize prior
        if pid not in player_state:
            player_state[pid] = {
                "mean": 5.0,   # neutral prior
                "var": 4.0     # high uncertainty
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
            "bayesian_uncertainty": round(new_var, 3)
        })

    return pd.DataFrame(results)

