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

df = pd.read_csv("../processed/final_labeled_data.csv")
coords = generate_coordinates_csv(df, "coordinates.csv")
coords.head()