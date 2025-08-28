import torch
import torch.nn as nn

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=padding, count_include_pad=False)

    def forward(self, x):  # x: [B, C, L]
        trend = self.moving_avg(x)
        res = x - trend
        return res, trend

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels=2, user_embeds=0, user_vocab=1):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.user_embeds = user_embeds

        self.decomp = SeriesDecomp(kernel_size=25)

        # 动态计算输入维度
        in_features = 2 * in_channels * seq_len  # seasonal + trend
        if user_embeds and user_vocab > 1:
            in_features += user_embeds

        self.linear_seasonal = nn.Linear(in_features, pred_len)
        self.linear_trend = nn.Linear(in_features, pred_len)

        if user_embeds and user_vocab > 1:
            self.user_embed = nn.Embedding(user_vocab, user_embeds)
        else:
            self.user_embed = None

    def forward(self, x, user_idx):  # x: [B, L, C]
        B, L, C = x.shape
        xt = x.transpose(1, 2)  # [B, C, L]
        seasonal, trend = self.decomp(xt)
        base = torch.cat([seasonal, trend], dim=1).contiguous().view(B, -1)  # [B, 2*C*L]

        if self.user_embed is not None:
            ue = self.user_embed(user_idx)  # [B, U]
            base = torch.cat([base, ue], dim=1)

        # 可选：打印调试
        # print("base.shape:", base.shape)

        y_seasonal = self.linear_seasonal(base)
        y_trend = self.linear_trend(base)
        y = y_seasonal + y_trend  # [B, pred_len]
        return y
