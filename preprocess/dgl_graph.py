# make csv file to DGL format
import dgl
import torch
import torch.nn.functional as F
import pandas as pd
import os

data_original_dir = './datasets/cities/original'
data_dir = './datasets/cities/'
data_norm_dir = './datasets/cities/norm'

def make_dgl(city_name):
    nodes_data = pd.read_csv("../datasets/cities/norm/atlanta_node_norm.csv")
    edges_data = pd.read_csv("../datasets/cities/norm/atlanta_edge_norm.csv")

    src = edges_data['src'].to_numpy()
    dst = edges_data['dst'].to_numpy()

    g = dgl.graph((src, dst))

    lon = torch.tensor(nodes_data['lon'].to_numpy()).float()
    g.ndata['lon'] = lon
    lat = torch.tensor(nodes_data['lat'].to_numpy()).float()
    g.ndata['lat'] = lat

    edge_weight = torch.tensor(edges_data['weight'].to_numpy())
    g.edata['weight'] = edge_weight

    return g
