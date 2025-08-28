
import torch
import torch.nn as nn

class MixerBlock(nn.Module):
    def __init__(self, seq_len, hidden_dim, feature_dim):
        super().__init__()
        self.norm_t = nn.LayerNorm(feature_dim)
        self.fc_t1 = nn.Linear(seq_len, hidden_dim)
        self.fc_t2 = nn.Linear(hidden_dim, seq_len)
        self.norm_c = nn.LayerNorm(feature_dim)
        self.fc_c1 = nn.Linear(feature_dim, hidden_dim)
        self.fc_c2 = nn.Linear(hidden_dim, feature_dim)
        self.act = nn.GELU()

    def forward(self, x):  # x: [B, L, C]
        # time mixing
        xt = self.norm_t(x)
        xt = xt.transpose(1,2)   # [B, C, L]
        xt = self.fc_t2(self.act(self.fc_t1(xt)))
        xt = xt.transpose(1,2)
        x = x + xt
        # channel mixing
        xc = self.norm_c(x)
        xc = self.fc_c2(self.act(self.fc_c1(xc)))
        x = x + xc
        return x

class TSMixer(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels=2, depth=4, hidden=128, user_embeds=0, user_vocab=1):
        super().__init__()
        self.embed = nn.Linear(in_channels, hidden)
        self.blocks = nn.ModuleList([MixerBlock(seq_len, hidden, hidden) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, pred_len)
        )
        if user_embeds>0:
            self.user_embed = nn.Embedding(user_vocab, user_embeds)
            self.ue_proj = nn.Linear(user_embeds, hidden)
        else:
            self.user_embed = None

    def forward(self, x, user_idx):  # x: [B, L, C]
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        # pool over time (mean)
        h = h.mean(dim=1)  # [B, H]
        if self.user_embed is not None:
            ue = self.ue_proj(self.user_embed(user_idx))
            h = h + ue
        y = self.head(h)  # [B, pred_len]
        return y
