import math
import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch.nn.functional as F
from .__base__ import BaseDGN


class GCN(BaseDGN):

    def __init__(self, **kwargs):

        super(GCN, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        params.pop('layer')
        conv = geom_nn.GCNConv(**params)
        return conv, conv.out_channels


class GCN2(BaseDGN):

    def __init__(self, **kwargs):
        super(GCN2, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        in_ch = params.pop('in_channels')
        out_ch = params.pop('out_channels')
        assert in_ch == out_ch
        params['channels'] = in_ch
        conv = geom_nn.GCN2Conv(**params)
        return conv, conv.channels


class GATv2(BaseDGN):

    def __init__(self, **kwargs):
        super(GATv2, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        expected_out_channels = params['out_channels']
        n_heads = params['heads']

        true_out_channels = int(math.ceil(expected_out_channels / n_heads))
        params['out_channels'] = true_out_channels

        return geom_nn.GATv2Conv(**params), true_out_channels*n_heads

    def forward(self, data, **other_parms):
        return super(GATv2, self).forward(data, return_attention_weights=True, **other_parms)

    def __extract_conv_results__(self, conv_results):
        return conv_results[0], conv_results[1][1]


class GraphSAGE(BaseDGN):

    def __init__(self, **kwargs):
        super(GraphSAGE, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        conv = geom_nn.SAGEConv(**params)
        return conv, conv.out_channels


class PNA(BaseDGN):

    def __init__(self, **kwargs):
        super(PNA, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        conv = geom_nn.PNAConv(**params)
        return conv, conv.out_channels


class GIN(BaseDGN):

    def __init__(self, **kwargs):
        super(GIN, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        in_ch = params.pop('in_channels')
        out_ch = params.pop('out_channels')
        phi = nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch))
        conv = geom_nn.GINConv(nn=phi, **params)
        return conv, out_ch


class AntisymmetricGNN(BaseDGN):

    def __init__(self, **kwargs):
        super(AntisymmetricGNN, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        params.pop('out_channels')
        params.pop('add_self_loops')

        conv = geom_nn.AntiSymmetricConv(**params)
        return conv, conv.in_channels


class MLP(BaseDGN):

    class MyMLPLayer(nn.Module):
        def __init__(self, in_channels, out_channels, act=F.relu, **other_params):
            super().__init__()
            self.out_channels = out_channels
            self.in_channels = in_channels
            self.l = nn.Linear(in_channels, out_channels)
            self.act = act

        def forward(self, layer_input, edge_index, **other_parms):
            return self.act(self.l(layer_input))

    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

    def __init_conv__(self, **params):
        params.pop('add_self_loops')

        conv = MLP.MyMLPLayer(**params)
        return conv, conv.out_channels
