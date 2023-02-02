import sys
import copy
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from torch import nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree





# MLP
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=256, bias=True, dim_inner=256, num_layers=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_inner),
            nn.ReLU(),
            nn.Linear(dim_inner, dim_out),
            nn.ReLU()
        )

    def forward(self, input):
        z = self.model(input)
        return z


# general conv
class GeneralConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=True)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.activation=nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)
        x=self.activation(x)


        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('int')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
