import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, batch_size, y_dim, x_dim, feat_dim,
                 temperature=10000, normalize=False, scale=None, device='cpu'):
        super().__init__()

        if scale is None:
            scale = 2 * math.pi

        y_embed, x_embed = torch.meshgrid(torch.linspace(1, y_dim, y_dim, device=device),
                                          torch.linspace(1, x_dim, x_dim, device=device))

        y_embed = y_embed.expand(batch_size, y_dim, x_dim)
        x_embed = x_embed.expand(batch_size, y_dim, x_dim)

        # print("x_meshgrid:", x_embed[0])

        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        # print("normalized x_meshgrid:", x_embed[0])

        dim_t = torch.arange(feat_dim, device=device)
        dim_t = temperature ** (2 * (dim_t // 2) / feat_dim)

        # print("dim_t:", dim_t)

        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, None] / dim_t

        # print("pos_x size:", pos_x.size())
        # print("pos_x:", pos_x[0])

        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # print("stacked pos_x size:", pos_x.size())

        self.pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        # print("pos size:", self.pos.size())

    def forward(self, x):
        return x + self.pos


class PositionalEmbedding(nn.Module):
    def __init__(self, n_controlpoints, d_model=256):
        super().__init__()
        self.embed = nn.Embedding(50, d_model)
        self.n_controlpoints = n_controlpoints

    def forward(self, x):
        b, hw, c = x.size()

        i = torch.arange(self.n_controlpoints, device=x.device)
        tgt = self.embed(i).repeat(b, 1, 1)
        return tgt
