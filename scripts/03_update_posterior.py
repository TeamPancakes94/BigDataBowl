# 03_update_posterior.py
# Bayesian beta–binomial for player, side, pillar and overall score 0–100

# placement 

import pandas as pd
import numpy as np
from scipy.stats import beta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
OUT_DIR = OUT / "posterior"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILLARS_CSV = OUT / "merged_pillars.csv"
CORE_CSV    = OUT / "merged_core.csv"

feat_df = pd.read_csv(PILLARS_CSV, low_memory=False)

assert PILLARS_CSV.exists(), f"Missing {PILLARS_CSV}"
assert CORE_CSV.exists(),    f"Missing {CORE_CSV}"

pillars = pd.read_csv(PILLARS_CSV, low_memory=False)
core    = pd.read_csv(CORE_CSV,    low_memory=False)

WR_WIN = {
    "anticipation": 120.0, 
    "execution":    0.50,  
    "separation":   1.50,  
    "innovation":   10.0,  
    # no 'eyes' here because eyes is a 1–10 score in script 02
}
DB_WIN = {
    "reaction": 140.0,      
    "coverage": 1.25,       
    # If you later compute DB 'eyes' (1–10), add a separate threshold for DB
}

# eyes threshold on a 1–10 scale 
EYES_THRESH_10 = 8.0

# cverall weights
WR_W = {"anticipation":0.30, "execution":0.20, "separation":0.20, "innovation":0.15, "eyes":0.15}
DB_W = {"coverage":0.50, "reaction":0.50}  
# add "eyes":0.10 later if you compute DB eyes

VALID_PILLARS = set(WR_W) | set(DB_W)


def pillar_success(side: str, pillar: str, v: float) -> int:
    if pd.isna(v):
        return 0
    if side == "WR":
        if pillar == "anticipation": return int(v <= WR_WIN["anticipation"])
        if pillar == "execution":    return int(v <= WR_WIN["execution"])
        if pillar == "separation":   return int(v >= WR_WIN["separation"])
        if pillar == "innovation":   return int(v >= WR_WIN["innovation"])
        if pillar == "eyes":         return int(v >= EYES_THRESH_10)   # <-- Eyes is 1–10
    else:
        if pillar == "reaction":     return int(v <= DB_WIN["reaction"])
        if pillar == "coverage":     return int(v <= DB_WIN["coverage"])
        # If you later compute DB eyes as 1–10, you could add:
        # if pillar == "eyes":       return int(v >= EYES_THRESH_10)
    return 0

def beta_update(s: int, f: int, a0: float = 5.0, b0: float = 5.0):
    a = a0 + s
    b = b0 + f
    mean = a / (a + b) if (a + b) > 0 else np.nan
    lo, hi = (beta.ppf([0.025, 0.975], a, b) if a > 0 and b > 0 else (np.nan, np.nan))
    return a, b, mean, lo, hi

def overall(side: str, means: dict) -> float:
    W = WR_W if side == "WR" else DB_W
    # only include pillars that exist; renormalize weights; scale to 0–100
    avail = {k: w for k, w in W.items() if k in means and pd.notna(means[k])}
    if not avail:
        return np.nan
    Z = sum(avail.values())
    return round(100.0 * sum((w / Z) * means[k] for k, w in avail.items()), 1)

def main():
    df = feat_df.copy()

    # Handle either snake_case (preferred) or legacy camelCase from older scripts
    rename = {}
    if "playerId" in df.columns: rename["playerId"] = "player_id"
    if "gameId"   in df.columns: rename["gameId"]   = "game_id"
    if "playId"   in df.columns: rename["playId"]   = "play_id"
    if rename:
        df = df.rename(columns=rename)

    # keep only pillars we understand and ensure numeric values
    df = df[df["pillar"].isin(VALID_PILLARS)].copy()
    df["raw_value"] = pd.to_numeric(df["raw_value"], errors="coerce")

    # success flag per row
    df["success"] = df.apply(lambda r: pillar_success(r["side"], r["pillar"], r["raw_value"]), axis=1)

    # aggregate per player × side × pillar
    g = df.groupby(["player_id", "side", "pillar"])["success"].agg(["sum", "count"]).reset_index()
    g["fails"] = g["count"] - g["sum"]

    # posterior per pillar
    rows = []
    for _, r in g.iterrows():
        a, b, mean, lo, hi = beta_update(int(r["sum"]), int(r["fails"]))
        rows.append({
            "player_id": r["player_id"],
            "side": r["side"],
            "pillar": r["pillar"],
            "alpha": a,
            "beta": b,
            "mean": mean,
            "score_1_10": round(10 * mean) if pd.notna(mean) else np.nan,
            "ci_low": lo,
            "ci_high": hi,
        })
    pillars_post = pd.DataFrame(rows)
    pillars_post.to_csv(OUT_DIR / "posterior_pillars.csv", index=False)


    # overall per player × side
    outs = []
    for (pid, side), sub in pillars_post.groupby(["player_id", "side"]):
        means = {row["pillar"]: row["mean"] for _, row in sub.iterrows()}
        W = WR_W if side == "WR" else DB_W
        pill_vec = {f"pillar_{k}": means.get(k, np.nan) for k in W}

        # --- compute PER-10 for WRs only ---
        per10 = np.nan
        if side == "WR":
            A = means.get("anticipation", np.nan)
            S = means.get("separation", np.nan)
            E = means.get("execution", np.nan)
            Eyes = means.get("eyes", np.nan)
            Innovation = means.get("innovation", np.nan)
            per10 = round(np.nanmean([A, S, E, Eyes, Innovation]) * 10, 1)
        # --- end PER-10 ---

        outs.append({
            "player_id": pid,
            "side": side,
            **pill_vec,
            "overall_0_100": overall(side, means),
            "per10": per10
        })


    overall_df = pd.DataFrame(outs)
    overall_df.to_csv(OUT_DIR / "posterior_overall.csv", index=False)

    print("Wrote:", (OUT_DIR / "posterior_pillars.csv").resolve())
    print("Wrote:", (OUT_DIR / "posterior_overall.csv").resolve())

if __name__ == "__main__":
    main()