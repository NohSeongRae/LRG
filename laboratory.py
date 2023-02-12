import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
from plot import plot_result
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from simple_data import make_simple_dataset
from spektral.datasets import citation
from spektral.layers import ops
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from simple_data import make_simple_dataset
from blocks import NDP
from spektral.utils import sparse
from etc.utils import to_numpy

#plot_result()
# A, X, _ =make_simple_dataset("Grid2d")
# print(type(A))
# A=sp_matrix_to_sp_tensor(A)
# diag_deg, _ = np.histogram(A.nonzero()[0], np.arange(A.shape[0]+1))
# print(len(diag_deg))
# dim = A.shape[0]
# diag_mat = np.zeros((dim**2, ))
# diag_mat[np.arange(0, dim**2, dim+1)] = diag_deg
# diag_mat.reshape((dim, dim))
# print(diag_mat)
# print(diag_mat.shape)