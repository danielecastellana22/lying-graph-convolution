import torch as th
import torch.nn as nn


class DiagonalEdgeWeightBuilder(nn.Module):

    def __init__(self, in_channels, out_channels, symmetric):
        super(DiagonalEdgeWeightBuilder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.symmetric = symmetric

        self.mlp = nn.Sequential(nn.Linear(2 * in_channels, out_channels, bias=False), nn.Tanh())

    def forward(self, x_u, x_v):
        out = self.mlp(th.cat([x_u, x_v], dim=1))

        if self.symmetric:
            out_2 = self.mlp(th.cat([x_v, x_u], dim=1))
            out = out * out_2

        return out