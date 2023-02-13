import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse

def data_loader(city_name):
    node_file_path = "../datasets/cities/norm/" + city_name + "_node_norm_highway_one.csv"
    edge_file_path = "../datasets/cities/norm/" + city_name + "_edge_norm_ver1_highway.csv"

    node = pd.read_csv('C:/Users/rlaqhdrb/Desktop/Data/norm/firenze_node_norm_highway_one.csv')
    node_id = node['id']
    node_lon = node['lon']
    node_lat = node['lat']
    node_highway = node['highway']

    edge = pd.read_csv('C:/Users/rlaqhdrb/Desktop/Data/norm/firenze_edge_norm_ver1.csv')
    edge_src = edge['src']
    edge_dst = edge['dst']

    G = nx.Graph()

    # Node
    node_list = []

    for i in range(len(node_lon)):
        node_list.append([node_lon[i], node_lat[i]])

    x = np.array(node_list)

    # Edge
    for i in range(len(edge_src)):
        G.add_edge(edge_src[i], edge_dst[i])

    A = nx.adjacency_matrix(G)
    A = sparse.lil_matrix(A)

    return A, x
