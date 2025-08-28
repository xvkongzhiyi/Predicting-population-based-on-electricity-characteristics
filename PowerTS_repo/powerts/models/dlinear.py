
import torch
import torch.nn as nn

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        padding = (kernel_size-1)//2
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=padding, count_include_pad=False)

    def forward(self, x):  # x: [B, C, L]
        trend = self.moving_avg(x)
        res = x - trend
        return res, trend

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels=2, user_embeds=0, user_vocab=1):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=25)
        c = in_channels + (user_embeds>0)
        self.linear_seasonal = nn.Linear(seq_len*c, pred_len)
        self.linear_trend = nn.Linear(seq_len*c, pred_len)
        if user_embeds>0:
            self.user_embed = nn.Embedding(user_vocab, user_embeds)
        else:
            self.user_embed = None

    def forward(self, x, user_idx):  # x: [B, L, C]
        B, L, C = x.shape
        xt = x.transpose(1,2)  # [B, C, L]
        seasonal, trend = self.decomp(xt)
        # concat channels and optional user embedding (tiled over time then flattened)
        base = torch.cat([seasonal, trend], dim=1).contiguous().view(B, -1)  # [B, 2C*L]
        if self.user_embed is not None:
            ue = self.user_embed(user_idx)  # [B, U]
            # tile across time "L" and append
            ue_tiled = ue
            base = torch.cat([base, ue_tiled], dim=1)
        y_seasonal = self.linear_seasonal(base)
        y_trend = self.linear_trend(base)
        y = y_seasonal + y_trend  # [B, pred_len]
        return y
