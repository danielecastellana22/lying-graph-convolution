from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import add_remaining_self_loops
from .edge_weight_builder import DiagonalEdgeWeightBuilder


class LyingGCNConv(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, symmetric=False, **kwargs):

        # copied from GCN
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        # to build the edge weight
        self.edge_w_builder = DiagonalEdgeWeightBuilder(in_channels, out_channels, symmetric)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tuple[Tensor, Tensor]:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --------------------------------------------------------------------------------------------------------------
        # sheaf weight should not be with self-loops
        sheaf_weight_to_store = self.edge_w_builder(x[edge_index[0]], x[edge_index[1]])  # E x out_channels

        if self.add_self_loops:
            edge_index, sheaf_weight = add_remaining_self_loops(edge_index, sheaf_weight_to_store, 1,
                                                                x.size(self.node_dim))
        else:
            sheaf_weight = sheaf_weight_to_store

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # TODO: this works only with edge_index! We should build sparse martix to work also with adj_t
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight.reshape(-1, 1) * sheaf_weight, size=None)
        # --------------------------------------------------------------------------------------------------------------

        if self.bias is not None:
            out = out + self.bias

        return out, sheaf_weight_to_store

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        raise NotImplementedError()
        # return spmm(adj_t, x, reduce=self.aggr)
