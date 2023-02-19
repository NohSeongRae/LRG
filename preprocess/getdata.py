import networkx as nx
import osmnx as ox
import pandas as pd
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
import os

ox.config(use_cache=True, log_console=True)
ox.__version__

## info.csv

city_name = "Firenze"
city_name_file = (city_name.lower()).replace(" ", "")
node_info_file_name = "../datasets/cities/info/" + city_name_file + "_node_info.csv"
edge_info_file_name = "../datasets/cities/info/" + city_name_file + "_edge_info.csv"

cf3 = '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]'
G = ox.graph_from_place(city_name, network_type="all", custom_filter=cf3)
G = ox.get_undirected(G)

fig_file_name = "../datasets/cities/" + city_name_file + ".png"
fig, ax = ox.plot_graph(G, node_size=1, edge_linewidth=0.5, save=True, filepath=fig_file_name)
nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

df_nodes = pd.DataFrame(nodes)
df_edges = pd.DataFrame(edges)
df_nodes.to_csv(node_info_file_name)
df_edges.to_csv(edge_info_file_name)

## original.csv

node_original_file_name = "../datasets/cities/original/" +city_name_file + "_node_original.csv"
edge_original_file_name = "../datasets/cities/original/" + city_name_file + "_edge_original.csv"

df_nodes = pd.read_csv(node_info_file_name)
node_id = df_nodes['osmid']
node_lon = df_nodes['x']
node_lat = df_nodes['y']

df_edges = pd.read_csv(edge_info_file_name)
edge_src = df_edges['u']
print(len(edge_src))
edge_dst = df_edges['v']
edge_len = df_edges['length']
highway_info = df_edges['highway']

index1 = []

for i in range (len(node_lon)):
    index1.append(i)

node_df = pd.DataFrame(index1, columns=['id'])
node_df['id'] = node_id
node_df['lon'] = node_lon
node_df['lat'] = node_lat

node_df.to_csv(node_original_file_name, index = False)

index2 = []

for i in range (len(edge_src)):
    index2.append(i)

edge_df = pd.DataFrame(index2, columns=['src'])
edge_df['src'] = edge_src
edge_df['dst'] = edge_dst
edge_df['weight'] = edge_len
edge_df['highway'] = highway_info

edge_df.to_csv(edge_original_file_name, index=False)

## norm.csv

atlanta_node = pd.read_csv(node_original_file_name)

node_id = list(atlanta_node['id'])
node_lon = list(atlanta_node['lon'])
node_lat = list(atlanta_node['lat'])

atlanta_edge = pd.read_csv(edge_original_file_name)

edge_src = list(atlanta_edge['src'])
edge_dst = list(atlanta_edge['dst'])
edge_length = list(atlanta_edge['weight'])
edge_highway = list(atlanta_edge['highway'])

self_loop = []

for i in range(len(edge_src)):
    if edge_src[i] == edge_dst[i]:
        self_loop.append(i)

k = 0
for i in range(len(self_loop)):
    del edge_length[self_loop[i] - k]
    del edge_src[self_loop[i] - k]
    del edge_dst[self_loop[i] - k]
    del edge_highway[self_loop[i] - k]
    k += 1

src_dst = []
multi_edge = []

for i in range(len(edge_src)):
    src_dst.append((edge_src[i], edge_dst[i]))

for j in range(len(src_dst) - 1):
    if src_dst[j] == src_dst[j + 1]:
        multi_edge.append(j + 1)

k = 0
for i in range(len(multi_edge)):
    del edge_length[multi_edge[i] - k]
    del edge_src[multi_edge[i] - k]
    del edge_dst[multi_edge[i] - k]
    del edge_highway[multi_edge[i] - k]
    k += 1

no_edge = []

for i in range(len(node_id)):
    if (node_id[i] not in edge_src) and (node_id[i] not in edge_dst):
        no_edge.append(i)

k = 0
for i in range(len(no_edge)):
    del node_id[no_edge[i] - k]
    del node_lon[no_edge[i] - k]
    del node_lat[no_edge[i] - k]
    k += 1

node_norm_file_name = "../datasets/cities/norm/" + city_name_file + "_node_norm.csv"
edge_norm_file_name = "../datasets/cities/norm/" + city_name_file + "_edge_norm_ver1_highway.csv"

node_df = pd.DataFrame(node_id, columns=['id'])
node_df['lon'] = node_lon
node_df['lat'] = node_lat

node_df.to_csv(node_norm_file_name, index=False)

edge_df = pd.DataFrame(edge_src, columns=['src'])
edge_df['dst'] = edge_dst
edge_df['weight'] = edge_length
edge_df['highway'] = edge_highway

edge_df.to_csv(edge_norm_file_name, index=False)

## norm_highway.csv

edge_data = pd.read_csv(edge_norm_file_name)
highway = edge_data['highway']
src = edge_data['src']
dst = edge_data['dst']

node_data = pd.read_csv(node_norm_file_name)
node_id = node_data['id']
node_lon = node_data['lon']
node_lat = node_data['lat']

node_dict = {}

## 여기 아래부분 수정

print(node_id)

for i in range(len(node_id)):
    node_dict[node_id[i]] = [0]

for i in range(len(src)):
    node_dict[src[i]] = list(node_dict[src[i]] + [highway[i]])

for i in range(len(dst)):
    node_dict[dst[i]] = list(node_dict[dst[i]] + [highway[i]])

for i in range(len(node_id)):
    del node_dict[node_id[i]][0]

df = pd.DataFrame(node_id, columns=["id"])
df['lon'] = node_lon
df['lat'] = node_lat
df['highway'] = list(node_dict.values())

node_highway_file_name = "../datasets/cities/norm/" + city_name_file + "_node_norm_highway.csv"

df.to_csv(node_highway_file_name, index=False)

## one.csv

edge_data = pd.read_csv(edge_norm_file_name)
highway = list(edge_data['highway'])
src = list(edge_data['src'])
dst = list(edge_data['dst'])

node_data = pd.read_csv(node_norm_file_name)
node_id = list(node_data['id'])
node_lon = list(node_data['lon'])
node_lat = list(node_data['lat'])

node_dict = {}

for i in range(len(node_id)):
    node_dict[node_id[i]] = [0]

for i in range(len(src)):
    node_dict[src[i]] = node_dict[src[i]] + [highway[i]]

for i in range(len(dst)):
    node_dict[dst[i]] = node_dict[dst[i]] + [highway[i]]

for i in range(len(node_id)):
    del node_dict[node_id[i]][0]

df = pd.DataFrame(node_id, columns=["id"])
df['lon'] = node_lon
df['lat'] = node_lat
df['highway'] = list(node_dict.values())

df.to_csv(node_highway_file_name, index=False)

highway_list1 = ["motorway", "motorway_link", "trunk", "trunk_link", "['trunk_link', 'trunk']"]
highway_list2 = ["primary", "primary_link"]
highway_list3 = ["secondary", "secondary_link"]
highway_list4 = ["unclassified", "tertiary", "tertiary_link"]
highway_list5 = ["residential"]

highway1 = 0
highway2 = 0
highway3 = 0
highway4 = 0
highway5 = 0

highway_1 = []
highway_2 = []
highway_3 = []
highway_4 = []
highway_5 = []

highway_one = []

for i in range(len(node_dict)):
    for j in range(len(node_dict[node_id[i]])):
        if node_dict[node_id[i]][j] in highway_list1:
            highway_one.append([1, 0, 0, 0, 0])
            # highway1 += 1
        if node_dict[node_id[i]][j] in highway_list2:
            highway_one.append([0, 1, 0, 0, 0])
            # highway2 += 1
        if node_dict[node_id[i]][j] in highway_list3:
            highway_one.append([0, 0, 1, 0, 0])
            # highway3 += 1
        if node_dict[node_id[i]][j] in highway_list4:
            highway_one.append([0, 0, 0, 1, 0])
            # highway4 += 1
        if node_dict[node_id[i]][j] in highway_list5:
            highway_one.append([0, 0, 0, 0, 1])
            # highway5 += 1
    print(highway1)
    highway_1.append(highway1)
    highway_2.append(highway2)
    highway_3.append(highway3)
    highway_4.append(highway4)
    highway_5.append(highway5)
    # highway_list.append([highway1, highway2, highway3, highway4, highway5])
    highway1 = 0
    highway2 = 0
    highway3 = 0
    highway4 = 0
    highway5 = 0



df2 = pd.DataFrame(node_id, columns=["id"])
df2['lon'] = node_lon
df2['lat'] = node_lat
df2['highway1'] = highway_1
df2['highway2'] = highway_2
df2['highway3'] = highway_3
df2['highway4'] = highway_4
df2['highway5'] = highway_5

node_encoding_file_name = "../datasets/cities/norm/" + city_name_file + "_node_norm_highway_one.csv"

df2.to_csv(node_encoding_file_name, index=False)