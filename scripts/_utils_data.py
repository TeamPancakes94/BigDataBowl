# scripts/_utils_data.py

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"

def load_week(week: int, kind: str = "input") -> pd.DataFrame:
    """
    kind: 'input' or 'output'
    """
    pat = f"{kind}_2023_w{week:02d}.csv"
    df = pd.read_csv(TRAIN / pat, low_memory=False)
    # standardize cols once
    colmap = {"gameId":"game_id","playId":"play_id","nflId":"nfl_id",
              "frameId":"frame_id","playerRole":"player_role"}
    df = df.rename(columns={k:v for k,v in colmap.items() if k in df.columns})
    df.columns = [c.lower() for c in df.columns]  # normalize
    return df

def load_weeks(weeks, kind="input"):
    return pd.concat([load_week(w, kind) for w in weeks], ignore_index=True)
