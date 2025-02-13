import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):

    def __init__(self):
        super(Regressor, self).__init__()

        self.dim_red = nn.Sequential(nn.Linear(19121, 1024), nn.GELU())


        self.conv_init = nn.Sequential(nn.Conv1d(1, 16, kernel_size=2, stride=2),
                                       Residual(ConvBlock(16, 16, 1)),
                                       AttentionPool(16))

        dim_ins = [16, 8]
        dim_outs = [8, 4]
        conv_tower = []

        for dim_in, dim_out in zip(dim_ins, dim_outs):
            conv_tower.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=4),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out)
            ))

        self.conv_tower = nn.Sequential(*conv_tower)

        self.pre_mha_norm = nn.LayerNorm(816)
        self.mha = nn.MultiheadAttention(embed_dim=816, num_heads=24, dropout=0.1, batch_first=True)
        self.post_tf = Residual(
            nn.Sequential(
                nn.LayerNorm(816),
                nn.Linear(816, 1632),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(1632, 816),
                nn.Dropout(0.1)
            )
        )

        self.fc = nn.Sequential(nn.Linear(816, 32),
                                nn.Linear(32, 1))

    def forward(self, d1e, d2e, cl, gex):

        ge = self.dim_red(gex)
        d1e = d1e.squeeze(1)
        d2e = d2e.squeeze(1)

        #600 + 600 + 1024 + 1024 = 3248
        x = torch.cat([d1e, d2e, ge, cl], 1)

        x = x.unsqueeze(1)
        x = self.conv_init(x)
        x = self.conv_tower(x)
        x = x.flatten(1)
        x = x.unsqueeze(1)

        x = self.pre_mha_norm(x)
        x, _ = self.mha(x, x, x)
        x = self.post_tf(x)

        x = self.fc(x)
        x = x.squeeze(1)
        x = torch.flatten(x)

        return x


class AttentionPool(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.attn_logit = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, d, n = x.shape
        rem = n % 2

        if rem > 0:
            x = F.pad(x, (0, rem), value=0)
            x = x.view(b, d, -1, 2)

            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, rem), value=True)
            mask = mask.view(b, 1, -1, 2)

            logits = self.attn_logit(x)
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(mask, mask_value)

        else:
            x = x.view(b, d, -1, 2)
            logits = self.attn_logit(x)

        attn = logits.softmax(dim=-1)
        x = (x * attn).sum(dim=-1)

        return x


def ConvBlock(in_dim, out_dim, kernel_size=1):
    return nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.GELU(),
        nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2)
    )


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
