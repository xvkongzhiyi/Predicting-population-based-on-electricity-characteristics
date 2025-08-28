
import torch
import torch.nn as nn
import math

class TimesFMZeroShot(nn.Module):
    """A minimal interface that *mimics* plugging a foundation model.
    This offline version uses a simple seasonal naive (daily) as a fallback,
    so the pipeline runs without heavy dependencies.
    """
    def __init__(self, seq_len, pred_len, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x, user_idx):
        # x: [B, L, C], channel 0 is y_norm
        y_hist = x[...,0]
        if self.pred_len <= y_hist.shape[1]:
            y = y_hist[:, -self.pred_len:]
        else:
            # repeat last values
            last = y_hist[:, -1:]
            y = last.repeat(1, self.pred_len)
        return y
