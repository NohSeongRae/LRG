# Get road data from OpenStreetMap and save it as csv file

import networkx as nx
import osmnx as ox
import pandas as pd
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
import os

# ox.config(use_cache=True, log_console=True)
# ox.__version__

data_original_dir = './datasets/cities/original'
data_dir = './datasets/cities/'
data_norm_dir = './datasets/cities/norm'

def get_original_csv(city_name):
    G = ox.graph_from_place(city_name, network_type="all", simplify=False)
    G = ox.simplify_graph(G, strict=False)

    # fig, ax = ox.plot_graph(G, node_size=1, edge_linewidth=0.5)

    node_id = list(G.nodes)

    node_lon = []
    node_lat = []

    for i in range(len(node_id)):
        node_lon.append(G.nodes[node_id[i]]['x'])
        node_lat.append(G.nodes[node_id[i]]['y'])

    edge_list = []

    edge_src = []
    edge_dst = []

    for i in range(len(list(G.edges))):
        edge_src.append(list(G.edges)[i][0])
        edge_dst.append(list(G.edges)[i][1])

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges_series = edges['length']  # gives you a pandas series with edge lengths
    edge_length = list(edges_series)

    # Save

    node_df = pd.DataFrame(node_id, columns=['id'])
    node_df['lon'] = node_lon
    node_df['lat'] = node_lat

    node_file_name = city_name + "_node_original"
    node_save_dir = os.path.join(data_original_dir, node_file_name)
    node_df.to_csv(node_save_dir, index=False)

    edge_df = pd.DataFrame(edge_src, columns=['src'])
    edge_df['dst'] = edge_dst
    edge_df['weight'] = edge_length

    edge_file_name = city_name + "_edge_original"
    edge_save_dir = os.path.join(data_original_dir, edge_file_name)
    edge_df.to_csv(edge_save_dir, index=False)

