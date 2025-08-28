
import pandas as pd
import numpy as np

def load_and_preprocess(csv_path: str, norm: str = "standard"):
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    # basic checks
    expected = {"user_id","ds","temp","y"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # sort and fill per user hourly continuity
    df = df.sort_values(["user_id","ds"]).reset_index(drop=True)
    dfs = []
    for uid, g in df.groupby("user_id"):
        full_index = pd.date_range(g["ds"].min(), g["ds"].max(), freq="H")
        g = g.set_index("ds").reindex(full_index)
        g["user_id"] = uid
        # forward/back fill temp, keep y as NaN if missing
        g["temp"] = g["temp"].interpolate(limit_direction="both")
        dfs.append(g.reset_index().rename(columns={"index":"ds"}))
    df = pd.concat(dfs, ignore_index=True)
    # normalization per user for y and global for temp
    if norm == "standard":
        df["y_mean"] = df.groupby("user_id")["y"].transform("mean")
        df["y_std"] = df.groupby("user_id")["y"].transform("std").replace(0,1.0)
        df["y_norm"] = (df["y"] - df["y_mean"]) / df["y_std"]
        df["temp_mean"] = df["temp"].mean()
        df["temp_std"] = df["temp"].std() if df["temp"].std() > 0 else 1.0
        df["temp_norm"] = (df["temp"] - df["temp_mean"]) / df["temp_std"]
    else:
        df["y_norm"] = df["y"]
        df["temp_norm"] = df["temp"]
    return df

def train_valid_split(df: pd.DataFrame, valid_ratio: float = 0.2):
    # split by time per user and then concat
    parts = []
    for uid, g in df.groupby("user_id"):
        n = len(g)
        cut = int(n*(1-valid_ratio))
        parts.append((g.iloc[:cut], g.iloc[cut:]))
    train = pd.concat([p[0] for p in parts]).sort_values(["user_id","ds"]).reset_index(drop=True)
    valid = pd.concat([p[1] for p in parts]).sort_values(["user_id","ds"]).reset_index(drop=True)
    return train, valid
