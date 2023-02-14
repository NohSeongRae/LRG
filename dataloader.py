import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse

def data_loader(city_name):
    city_name = (city_name.lower()).replace(" ", "")

    node_file_path = "./datasets/cities/norm/" + city_name + "_node_norm_highway_one.csv"
    edge_file_path = "./datasets/cities/norm/" + city_name + "_edge_norm_ver1_highway.csv"

    node = pd.read_csv(node_file_path)
    node_id = node['id']
    node_lon = node['lon']
    node_lat = node['lat']
    node_highway1 = node['highway1']
    node_highway2 = node['highway2']
    node_highway3 = node['highway3']
    node_highway4 = node['highway4']
    node_highway5 = node['highway5']

    edge = pd.read_csv(edge_file_path)
    edge_src = edge['src']
    edge_dst = edge['dst']

    G = nx.Graph()

    node_1 = np.array([])
    node_2 = np.array([])
    node_3 = np.array([])
    node_4 = np.array([])
    node_5 = np.array([])

    # Node

    node_list1 = []
    node_list2 = []
    node_list3 = []
    node_list4 = []
    node_list5 = []


    for i in range(1, len(node_lon)):
        if node_highway1[i] != 0:
            node_arr_1_1 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.],
                                     [node_highway1[i], node_highway2[i], node_highway3[i], node_highway4[i],
                                      node_highway5[i]]])
            node_list1.append(node_arr_1_1)
        if node_highway2[i] != 0:
            node_arr_1_2 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.],
                                     [node_highway1[i], node_highway2[i], node_highway3[i], node_highway4[i],
                                      node_highway5[i]]])
            node_list2.append(node_arr_1_2)
        if node_highway3[i] != 0:
            node_arr_1_3 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.],
                                     [node_highway1[i], node_highway2[i], node_highway3[i], node_highway4[i],
                                      node_highway5[i]]])
            node_list3.append(node_arr_1_3)
        if node_highway4[i] != 0:
            node_arr_1_4 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.],
                                     [node_highway1[i], node_highway2[i], node_highway3[i], node_highway4[i],
                                      node_highway5[i]]])
            node_list4.append(node_arr_1_4)
        if node_highway5[i] != 0:
            node_arr_1_5 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.],
                                     [node_highway1[i], node_highway2[i], node_highway3[i], node_highway4[i],
                                      node_highway5[i]]])
            node_list5.append(node_arr_1_5)

    x1 = np.array(node_list1)
    x2 = np.array(node_list2)
    x3 = np.array(node_list3)
    x4 = np.array(node_list4)
    x5 = np.array(node_list5)


    # Edge
    for i in range(len(edge_src)):
        G.add_edge(edge_src[i], edge_dst[i])

    A = nx.adjacency_matrix(G)
    A = sparse.lil_matrix(A)

    return A, x1, x2, x3, x4, x5
