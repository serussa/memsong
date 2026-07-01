import torch
import torch.nn as nn


class Generator(nn.Module):
    """Audio quality evaluation model (5-dim regression head)."""

    def __init__(self, in_features=1024, ffd_hidden_size=4096, num_classes=5, attn_layer_num=4):
        super().__init__()
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=in_features, num_heads=8, batch_first=True)
            for _ in range(attn_layer_num)
        ])
        self.ffd = nn.Sequential(
            nn.Linear(in_features, ffd_hidden_size),
            nn.GELU(),
            nn.Linear(ffd_hidden_size, in_features),
        )
        self.fc = nn.Linear(in_features * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, in_features)
        for attn_layer in self.attn:
            attn_out, _ = attn_layer(x, x, x)
            x = x + attn_out

        ffd_out = self.ffd(x)
        x = x + ffd_out

        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1)[0]
        x = self.fc(torch.cat([mean_pool, max_pool], dim=-1))
        return x
