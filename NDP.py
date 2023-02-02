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




def NDP(A_list, level, sparsify=1e-2):
    A_out = []
    S_out = []
    for a in A_list:
        A_pool, S = _pool(a, level, sparsify=sparsify)
        A_out.append(A_pool)
        S_out.append(S)

    return A_out, S_out


def preprocess(X, A):
    return X, A


def _pool(A, level, sparsify=1e-2):
    masks = []
    A_out = A
    for i in range(level):
        A_out, mask = _select_and_connect(A_out)
        masks.append(mask)
    S = _masks_to_matrix(masks)

    A_out = A_out.tocsr()
    A_out = A_out.multiply(np.abs(A_out) > sparsify)
    return A_out, S


def _select_and_connect(A):
    A = sp.csc_matrix(A)
    L = laplacian(A)
    Ls = normalized_laplacian(A)

    if L.shape == (1, 1):
        idx_pos = np.zeros(1, dtype=int)
        V = np.ones(1)
    else:
        try:
            V = eigsh(Ls, k=1, which="LM", v0=np.ones(A.shape[0]))[1][:, 0]
        except Exception:
            print("Eigen-decomposition failed. Splitting nodes randomly instead")
            V = np.random.choice([-1, 1], size=(A.shape[0],))

        idx_pos = np.nonzero(V >= 0)[0]
        idx_neg = np.nonzero(V < 0)[0]

        z = np.ones((A.shape[0], 1))
        z[idx_neg] = -1
        cut_size = eval_cut(A, L, z)
        if cut_size < 0.5:
            print(
                "Spectral cut lower than 0.5 {}: returning random cut".format(cut_size)
            )
            V = np.random.choice([-1, 1], size=(Ls.shape[0],))
            idx_pos = np.nonzero(V >= 0)[0]
            idx_neg = np.nonzero(V < 0)[0]

    if len(idx_pos) <= 1:
        L_pool = sp.csc_matrix(np.zeros((1, 1)))
        idx_pos = np.zeros(1, dtype=int)

    else:
        L_pool = _kron_reduction(L, idx_pos, idx_neg)

    if np.abs(L_pool - L_pool.T).sum() < np.spacing(1) * np.abs(L_pool).sum():
        L_pool = (L_pool + L_pool.T) / 2.0

    A_pool = -L_pool
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A_pool.setdiag(0)
        A_pool.eliminate_zeros()

    mask = np.zeros_like(V, dtype=bool)
    mask[idx_pos] = True
    return A_pool, mask


def _kron_reduction(L, idx_pos, idx_neg):
    L_red = L[np.ix_(idx_pos, idx_pos)]
    L_in_out = L[np.ix_(idx_pos, idx_neg)]
    L_out_in = L[np.ix_(idx_neg, idx_pos)].tocsc()
    L_comp = L[np.ix_(idx_neg, idx_neg)].tocsc()
    try:
        L_pool = L_red - L_in_out.dot(sp.linalg.spsolve(L_comp, L_out_in))
    except RuntimeError:
        ml_c = sp.csc_matrix(sp.eye(L_comp.shape[0]) * 1e-6)
        L_pool = L_red - L_in_out.dot(sp.linalg.spsolve(ml_c + L_comp, L_out_in))

    return L_pool


def eval_cut(A, L, z):
    cut = z.T.dot(L.dot(z))
    cut /= 2 * np.sum(A)
    return cut


def _masks_to_matrix(masks):
    S_ = sp.eye(masks[0].shape[0], dtype=np.float32).tocsr()
    S_ = S_[:, masks[0]]
    for i in range(1, len(masks)):
        S_next = sp.eye(masks[i].shape[0], dtype=np.float32).tocsr()
        S_next = S_next[:, masks[i]]
        S_ = S_.dot(S_next)
    return S_


def degree_matrix(A):
    degrees = np.array(A.sum(1)).flatten()
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def degree_power(A, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def normalized_adjacency(A, symmetric=True):
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)


def laplacian(A):
    return degree_matrix(A) - A


def normalized_laplacian(A, symmetric=True):
    if sp.issparse(A):
        I = sp.eye(A.shape[-1], dtype=A.dtype)
    else:
        I = np.eye(A.shape[-1], dtype=A.dtype)
    normalized_adj = normalized_adjacency(A, symmetric=symmetric)
    return I - normalized_adj