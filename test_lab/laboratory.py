import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp

from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from simple_data import make_simple_dataset
from spektral.datasets import citation
from spektral.layers import GCNConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from simple_data import make_simple_dataset
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# t_1=tf.random.normal([4, 6])
# z_1=tf.random.normal([6])
# # print(tf.random.normal([4, 6]))
# # t_2=tf.transpose(t_1)
# # print(t_2)
# # t_3=tf.matmul(t_1, t_2)
# # print(t_3)
#
# print(t_1)
# print(z_1)
# print(t_1*z_1)
# # t_list=[1,2,3,4]
# # print(t_list[:2])

# dataset=citation.Cora()
# graph=dataset[0]
# X=graph.x
# A=graph.a
# print(f"graph: {graph}")
# # print(f"graph.x: {X}")
# print(f"len(graph.x): {len(X)}")
# print(f"graph.a: {A}")
# print("------------------------------------------------")
# print("------------------------------------------------")
# print(f"sp.eye(graph.a.shape[0], dtype=np.float32): {sp.eye(graph.a.shape[0], dtype=np.float32)}")
# print("------------------------------------------------")
# print("------------------------------------------------")
# A=A+sp.eye(graph.a.shape[0], dtype=np.float32)
# A=A.toarray().reshape([-1])
# print(f"toarray().reshape([-1]): {A}")
# print(f"len(A): {len(A)}")

A, X, _=make_simple_dataset("Grid2d")
# #
# # print(type(A))
# # print(A)
# A=sp_matrix_to_sp_tensor(A)
A_label=A.toarray().reshape([-1])
print(len(A_label))
print(len(X))
# print(type(A))
# print(A)

# sp=tf.sparse.from_dense([[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,1],[0,1,1,0,1],[0,0,1,1,0]])
# print(sp)
# print(type(sp))
# print(sp.dense_shape)



# plot_result()
