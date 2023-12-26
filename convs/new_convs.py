import torch as th
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.utils import get_laplacian, add_self_loops, add_remaining_self_loops, degree
from .edge_weight_builder import DiagonalEdgeWeightBuilder


class LyingConv(MessagePassing):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(LyingConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_self_loops = kwargs['add_self_loops']
        self.symmetric = kwargs['symmetric'] if 'symmetric' in kwargs else False

        self.edge_w_builder = DiagonalEdgeWeightBuilder(in_channels, out_channels, self.symmetric)

        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        bias = kwargs['bias'] if 'bias' in kwargs else True
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, **other_params):
        N = x.shape[0]

        # sheaf weight should not be with self-loops
        sheaf_weight_to_store = self.edge_w_builder(x[edge_index[0]], x[edge_index[1]])  # E x out_channels

        if self.add_self_loops:
            edge_index, sheaf_weight = add_remaining_self_loops(edge_index, sheaf_weight_to_store, 1, N)
        else:
            sheaf_weight = sheaf_weight_to_store

        row, col = edge_index[0], edge_index[1]
        deg = degree(edge_index[0], N)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight.unsqueeze(1)*sheaf_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        return out, sheaf_weight_to_store

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight * x_j


# TODO: is slower than the non-sparse one
class SparseLyingConv(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(SparseLyingConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_self_loops = kwargs['add_self_loops']

        self.edge_w_builder = nn.Sequential(nn.Linear(2 * in_channels, out_channels), nn.Tanh())

        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        bias = kwargs['bias'] if 'bias' in kwargs else True
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def __get_sheaf_indexes__(self, edge_index, n_channels):
        n_edges = edge_index.shape[1]
        sheaf_edge_indexes = th.repeat_interleave(edge_index, n_channels, dim=1)  # contains E x OUT indexes
        sheaf_channels_indexes = th.arange(n_channels).reshape(1,-1).repeat(2, n_edges)
        sheaf_edge_index = th.stack([sheaf_edge_indexes[0]*n_channels + sheaf_channels_indexes[0],
                                     sheaf_edge_indexes[1]*n_channels + sheaf_channels_indexes[1]], dim=0)
        return sheaf_edge_index

    def forward(self, x, edge_index, **other_params):
        N_V = x.shape[0]  # number of nodes
        N_E = edge_index.shape[1]  # number of edges
        OUT = self.out_channels  # number of out channels

        # sheaf weight should not be with self-loops
        sheaf_weights = self.edge_w_builder(th.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)) # E x out_channels
        sheaf_indexes = self.__get_sheaf_indexes__(edge_index, OUT)

        E = th.sparse_coo_tensor(sheaf_indexes, sheaf_weights.reshape(-1), size=(N_V*OUT, N_V*OUT))

        # build the N_V x OUT Laplacian
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index)
        # TODO: get_laplacian could be the wrong choice
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym')

        sheaf_lapl_indexes = self.__get_sheaf_indexes__(edge_index, OUT)
        L_nd = th.sparse_coo_tensor(sheaf_lapl_indexes, th.repeat_interleave(edge_weight, OUT),
                                    size=(N_V*OUT, N_V*OUT))

        I_nd = th.sparse_coo_tensor(th.arange(N_V*OUT).reshape(1,-1).repeat(2,1), th.ones(N_V*OUT))
        solver_matrix = I_nd - L_nd * (E+I_nd)
        solver_matrix = solver_matrix.coalesce()
        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        #out = self.propagate(solver_matrix, x=x)
        out = th.sparse.mm(solver_matrix, x.reshape(-1,1)).reshape(x.shape)

        if self.bias is not None:
            out = out + self.bias

        return out, sheaf_weights


# TODO: this is a kind of non-symmetric sheaf laplacian since we do not multiply transport map across edges
class __LyingConv_old_error__(MessagePassing):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(__LyingConv_old_error__, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_self_loops = kwargs['add_self_loops']

        self.edge_w_builder = nn.Sequential(nn.Linear(2*in_channels, out_channels), nn.Tanh())

        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        bias = kwargs['bias'] if 'bias' in kwargs else True
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, **other_params):

        # sheaf weight should not be with self-loops
        sheaf_weight = self.edge_w_builder(th.cat((x[edge_index[0]], x[edge_index[1]]), dim=1))  # E x out_channels

        if self.add_self_loops:
            edge_index, sheaf_weight = add_self_loops(edge_index, sheaf_weight, fill_value=1.0)

        # TODO: this is wrong because the edge_weight should be D^(-1/2)AD^(-1/2) instead of laplacian
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym')

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight.unsqueeze(1)*sheaf_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        return out, sheaf_weight

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight * x_j
