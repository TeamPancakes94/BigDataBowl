import pandas as pd

def generate_coordinates_csv(df, out_path="coordinates.csv"):
    """
    Generates vision-style coordinate detections from tracking data.
    This replaces true YOLO/MediaPipe since frames are synthetic.
    """

    rows = []

    for (_, r) in df.iterrows():

        # Player point
        rows.append({
            "play_id": r["play_id"],
            "frame_id": r["frame_id"],
            "nfl_id": r["nfl_id"],
            "object_type": "player",
            "x": r["x"],
            "y": r["y"],
            "source": "tracking"
        })

        # Ball landing point (if available)
        if not pd.isna(r.get("ball_land_x")):
            rows.append({
                "play_id": r["play_id"],
                "frame_id": r["frame_id"],
                "nfl_id": -1,
                "object_type": "ball",
                "x": r["ball_land_x"],
                "y": r["ball_land_y"],
                "source": "tracking"
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    return out_df

'''
all_df = pd.read_csv("processed/input_cleaned_w1_2_3.csv")
labeled_df = pd.read_csv("processed/final_labeled_data.csv")
labeled_plays = labeled_df["play_id"].unique()
df = all_df[all_df["play_id"].isin(labeled_plays)]
coords = generate_coordinates_csv(df, "coordinates.csv")
coords.head()
'''
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

all_df = pd.read_csv(ROOT / "processed/input_cleaned_w1_2_3.csv")
labeled_df = pd.read_csv(ROOT / "processed/final_labeled_data.csv")
labeled_plays = labeled_df["play_id"].unique()

df = all_df.copy()

# write into outputs/coordinates.csv
coords = generate_coordinates_csv(df, OUT / "coordinates.csv")
print("Wrote coordinates →", (OUT / "coordinates.csv").resolve())
