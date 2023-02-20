import networkx as nx
import pandas as pd
import numpy as np
from scipy import sparse

def get_graph2(city_name):
    city_name_file = (city_name.lower()).replace(" ", "")
    edge_norm_file_name = "./datasets/cities/norm/" + city_name_file + "_edge_norm_ver1_highway.csv"
    node_norm_file_name = "./datasets/cities/norm/" + city_name_file + "_node_norm.csv"
    node_highway_file_name = "./datasets/cities/norm/" + city_name_file + "_node_norm_highway.csv"

    edge_data = pd.read_csv(edge_norm_file_name) # atlanta_edge_norm_ver1_highway
    highway = list(edge_data['highway'])
    edge_src = list(edge_data['src'])
    edge_dst = list(edge_data['dst'])

    node_data = pd.read_csv(node_norm_file_name)
    node_id = list(node_data['id'])
    node_lon = list(node_data['lon'])
    node_lat = list(node_data['lat'])

    ## 0~ 으로 정렬하기
    node_index = []
    for i in range(len(node_id)):
        node_index.append(i)

    temp_dict = {}
    for i in range(len(node_index)):
        temp_dict[node_id[i]] = i

    edge_src_sort = edge_src
    edge_dst_sort = edge_dst

    for i in range(len(edge_src)):
        edge_src_sort[i] = temp_dict[edge_src[i]]
        edge_dst_sort[i] = temp_dict[edge_dst[i]]

    ## return
    edges = [edge_src_sort, edge_dst_sort]

    G = nx.Graph()

    G.add_nodes_from(node_index)

    for i in range(len(edge_src_sort)):
        G.add_edge(edge_src_sort[i], edge_dst_sort[i])

    A = nx.adjacency_matrix(G)
    A = sparse.lil_matrix(A)

    # one-hot encoding

    highway_dict = {}

    for i in range(len(node_index)):
        highway_dict[i] = [0]

    for i in range(len(edge_src_sort)):
        highway_dict[edge_src_sort[i]] = highway_dict[edge_src_sort[i]] + [highway[i]]

    for i in range(len(edge_dst_sort)):
        highway_dict[edge_dst_sort[i]] = highway_dict[edge_dst_sort[i]] + [highway[i]]

    for i in range(len(node_index)):
        del highway_dict[node_index[i]][0]

    df = pd.DataFrame(node_index, columns=["id"])
    df['lon'] = node_lon
    df['lat'] = node_lat
    df['highway'] = list(highway_dict.values())

    df.to_csv(node_highway_file_name, index=False)

    highway_list1 = ["motorway", "motorway_link", "trunk", "trunk_link", "['trunk_link', 'trunk']", "primary", "primary_link"]
    highway_list2 = ["secondary", "secondary_link"]
    highway_list3 = ["tertiary", "tertiary_link"]
    highway_list4 = ["unclassified"]
    highway_list5 = ["residential"]

    highway_one = []

    for i in range(len(highway_dict)):
        for j in range(len(highway_dict[node_index[i]])):
            if highway_dict[node_index[i]][j] in highway_list1:
                highway_one.append([1, 0, 0, 0, 0])
            if highway_dict[node_index[i]][j] in highway_list2:
                highway_one.append([0, 1, 0, 0, 0])
            if highway_dict[node_index[i]][j] in highway_list3:
                highway_one.append([0, 0, 1, 0, 0])
            if highway_dict[node_index[i]][j] in highway_list4:
                highway_one.append([0, 0, 0, 1, 0])
            if highway_dict[node_index[i]][j] in highway_list5:
                highway_one.append([0, 0, 0, 0, 1])

    # node_feature
    node_feature1 = []
    node_feature2 = []
    node_feature3 = []
    node_feature4 = []
    node_feature5 = []

    node_id1 = []
    node_id2 = []
    node_id3 = []
    node_id4 = []
    node_id5 = []

    node_feature = []

    for i in range(len(node_id)):
        if highway_one[i][0] != 0:
            node_arr_1 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_feature1.append(node_arr_1)
            node_feature.append(node_arr_1)
            node_id1.append(node_index[i])
        if highway_one[i][1] != 0:
            node_arr_2 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_feature2.append(node_arr_2)
            node_feature.append(node_arr_2)
            node_id2.append(node_index[i])
        if highway_one[i][2] != 0:
            node_arr_3 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_feature3.append(node_arr_3)
            node_feature.append(node_arr_3)
            node_id3.append(node_index[i])
        if highway_one[i][3] != 0:
            node_arr_4 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_feature4.append(node_arr_4)
            node_feature.append(node_arr_4)
            node_id4.append(node_index[i])
        if highway_one[i][4] != 0:
            node_arr_5 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_feature5.append(node_arr_5)
            node_feature.append(node_arr_5)
            node_id5.append(node_index[i])

    node_ids = [node_id1, node_id2, node_id3, node_id4, node_id5]
    node_feature1 = np.array(node_feature1)
    node_feature2 = np.array(node_feature2)
    node_feature3 = np.array(node_feature3)
    node_feature4 = np.array(node_feature4)
    node_feature5 = np.array(node_feature5)

    node_feature = np.array(node_feature)
    print("node_feature",  len(node_feature))
    print("G.nodes", len(G.nodes))


    return G, A, node_feature, node_ids

G, A, node_feature, node_ids = get_graph2("Little Rock")

print(node_feature[0])