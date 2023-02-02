import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn

from torch import Tensor
import torch.nn.functional as F
# from dgl.nn.pytorch.conv import GraphConv
# from dgl.nn.pytorch import linear

import torch_geometric
import torch_geometric.graphgym as gym

def upsampling_from_matrix(inputs):
    if len(inputs) == 4:
        X, A, I, S = inputs
    else:
        X, A, S = inputs
        I = None

    X_out = K.dot(S, X)
    A_out = ops.matmul_at_b_a(ops.transpose(S), A)
    output = [X_out, A_out]

    if I is not None:
        I_out = K.dot(S, K.cast(I[:, None], tf.float32))[:, 0]
        I_out = K.cast(I_out, tf.int32)
        output.append(I_out)

    return output
def upsampling_with_pinv(inputs):
    if len(inputs) == 4:
        X, A, I, S = inputs
    else:
        X, A, S = inputs
        I = None

    S = tf.transpose(tf.linalg.pinv(S))
    return upsampling_from_matrix([X, A, I, S])