import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugResponseEmbedder(nn.Module):

    def __init__(self):
        super(DrugResponseEmbedder, self).__init__()

        self.dim_reducer = GeneExpDimReducer()

    def forward(self, cl_a, cl_b):

        if cl_b is None:
            return self.dim_reducer(cl_a)
        else:
            ae = self.dim_reducer(cl_a)
            be = self.dim_reducer(cl_b)

        return ae, be


class GeneExpDimReducer(nn.Module):

    def __init__(self):
        super(GeneExpDimReducer, self).__init__()

        self.conv_init = nn.Sequential(nn.Conv1d(1, 16, kernel_size=10, stride=10),
                                       Residual(ConvBlock(16, 16, 1)),
                                       AttentionPool(16))

        dim_ins = [16, 4]
        dim_outs = [4, 1]
        conv_tower = []

        for dim_in, dim_out in zip(dim_ins, dim_outs):
            conv_tower.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=4),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out)
            ))

        self.conv_tower = nn.Sequential(*conv_tower)

        self.pre_mha_norm = nn.LayerNorm(240)
        self.mha = nn.MultiheadAttention(embed_dim=240, num_heads=16, dropout=0.1, batch_first=True)
        self.post_tf = Residual(
                    nn.Sequential(
                        nn.LayerNorm(240),
                        nn.Linear(240, 480),
                        nn.Dropout(0.1),
                        nn.GELU(),
                        nn.Linear(480, 240),
                        nn.Dropout(0.1)
                    )
        )

        self.fc = nn.Linear(240, 1024)

    def forward(self, exp):

        x = exp.unsqueeze(1)
        x = self.conv_init(x)
        x = self.conv_tower(x)

        x = self.pre_mha_norm(x)
        x, _ = self.mha(x, x, x)
        x = self.post_tf(x)

        x = self.fc(x)
        x = x.squeeze(1)

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
