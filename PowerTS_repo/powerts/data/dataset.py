
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SlidingWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, pred_len: int):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []
        # per user, build windows
        for uid, g in df.groupby("user_id"):
            g = g.sort_values("ds").reset_index(drop=True)
            y = g["y_norm"].values.astype(np.float32)
            temp = g["temp_norm"].values.astype(np.float32)
            for i in range(0, len(g) - (seq_len + pred_len) + 1):
                x_y = y[i:i+seq_len]
                x_temp = temp[i:i+seq_len]
                y_future = y[i+seq_len:i+seq_len+pred_len]
                self.samples.append((uid, x_y, x_temp, y_future))
        # map user ids to indices
        self.user2idx = {u:i for i,u in enumerate(sorted(df["user_id"].unique()))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, x_y, x_temp, y_future = self.samples[idx]
        uidx = self.user2idx[uid]
        x = torch.tensor(
            np.stack([x_y, x_temp], axis=-1), dtype=torch.float32
        )  # [seq_len, 2]
        u = torch.tensor(uidx, dtype=torch.long)
        y = torch.tensor(y_future, dtype=torch.float32)  # [pred_len]
        return x, u, y
