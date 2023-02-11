import glob
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
def plot_result():
    minnesota = np.load('result/NDP_Grid2d_matrices.npz')

    # 1 Original

    origin_pos_dict = {}

    for i in range(len(minnesota["X"])):
        origin_pos_dict[i] = tuple(minnesota["X"][i])

    origin_A = minnesota["A"]
    origin_G = nx.from_numpy_array(origin_A)
    origin_pos = origin_pos_dict
    nx.draw_networkx_nodes(origin_G, origin_pos, node_size=2, node_color='black')
    nx.draw_networkx_edges(origin_G, origin_pos, alpha=0.5, width=1)
    plt.axis('off')
    plt.show()

    # 2 Predict

    pred_pos_dict = {}

    for i in range(len(minnesota["X_pred"])):
        pred_pos_dict[i] = tuple(minnesota["X_pred"][i])

    pred_A = minnesota["A_pred"]
    pred_G = nx.from_numpy_array(pred_A)
    pred_pos = pred_pos_dict
    nx.draw_networkx_nodes(pred_G, pred_pos, node_size=2, node_color='black')
    nx.draw_networkx_edges(pred_G, pred_pos, alpha=0.5, width=1)
    plt.axis('off')
    plt.show()
