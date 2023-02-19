import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
from scipy import sparse


def get_graph(city_name):
    # city_name = "Little Rock"
    city_name_file = (city_name.lower()).replace(" ", "")
    edge_norm_file_name = "./datasets/cities/norm/" + city_name_file + "_edge_norm_ver1_highway.csv"
    node_norm_file_name = "./datasets/cities/norm/" + city_name_file + "_node_norm.csv"
    node_highway_file_name = "./datasets/cities/norm/" + city_name_file + "_node_norm_highway.csv"
    edge_file_path = "./datasets/cities/norm/" + city_name_file + "_edge_norm_ver1_highway.csv"

    edge_data = pd.read_csv(edge_norm_file_name)
    highway = list(edge_data['highway'])
    src = list(edge_data['src'])
    dst = list(edge_data['dst'])

    node_data = pd.read_csv(node_norm_file_name)
    node_id = list(node_data['id'])
    node_lon = list(node_data['lon'])
    node_lat = list(node_data['lat'])

    node_index = []
    for i in range(len(node_id)):
        node_index.append(i)

    edge = pd.read_csv(edge_file_path)
    edge_src = edge['src']
    edge_dst = edge['dst']

    origin_G = nx.Graph()

    origin_G.add_nodes_from(node_id)

    for i in range(len(edge_src)):
        origin_G.add_edge(edge_src[i], edge_dst[i])

    origin_A = nx.adjacency_matrix(origin_G)
    origin_A = sparse.lil_matrix(origin_A)

    temp_dict = {}
    for i in range(len(node_index)):
        temp_dict[node_id[i]] = i
        node_index[i] = i

    for i in range(len(edge_src)):
        edge_src[i] = temp_dict[edge_src[i]]
        edge_dst[i] = temp_dict[edge_dst[i]]

    for i in range(len(edge_dst)):
        src[i] = temp_dict[src[i]]
        dst[i] = temp_dict[dst[i]]

    node_dict = {}

    for i in range(len(node_index)):
        node_dict[node_index[i]] = [0]

    for i in range(len(src)):
        node_dict[src[i]] = node_dict[src[i]] + [highway[i]]

    for i in range(len(dst)):
        node_dict[dst[i]] = node_dict[dst[i]] + [highway[i]]

    for i in range(len(node_index)):
        del node_dict[node_index[i]][0]

    df = pd.DataFrame(node_index, columns=["id"])
    df['lon'] = node_lon
    df['lat'] = node_lat
    df['highway'] = list(node_dict.values())

    df.to_csv(node_highway_file_name, index=False)

    highway_list1 = ["motorway", "motorway_link", "trunk", "trunk_link", "['trunk_link', 'trunk']", "primary", "primary_link"]
    highway_list2 = ["secondary", "secondary_link"]
    highway_list3 = ["tertiary", "tertiary_link"]
    highway_list4 = ["unclassified"]
    highway_list5 = ["residential"]

    highway_one = []

    # one-hot encoding

    for i in range(len(node_dict)):
        for j in range(len(node_dict[node_index[i]])):
            if node_dict[node_index[i]][j] in highway_list1:
                highway_one.append([1, 0, 0, 0, 0])
                # highway1 += 1
            if node_dict[node_index[i]][j] in highway_list2:
                highway_one.append([0, 1, 0, 0, 0])
                # highway2 += 1
            if node_dict[node_index[i]][j] in highway_list3:
                highway_one.append([0, 0, 1, 0, 0])
                # highway3 += 1
            if node_dict[node_index[i]][j] in highway_list4:
                highway_one.append([0, 0, 0, 1, 0])
                # highway4 += 1
            if node_dict[node_index[i]][j] in highway_list5:
                highway_one.append([0, 0, 0, 0, 1])
                # highway5 += 1

    node_list1 = []
    node_list2 = []
    node_list3 = []
    node_list4 = []
    node_list5 = []

    node_id1 = []
    node_id2 = []
    node_id3 = []
    node_id4 = []
    node_id5 = []

    edge_src1 = []
    edge_src2 = []
    edge_src3 = []
    edge_src4 = []
    edge_src5 = []

    edge_dst1 = []
    edge_dst2 = []
    edge_dst3 = []
    edge_dst4 = []
    edge_dst5 = []

    origin_order1 = []
    origin_order2 = []
    origin_order3 = []
    origin_order4 = []
    origin_order5 = []


    for i in range(1, len(node_lon)):
        if highway_one[i][0] != 0:
            node_arr_1 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_list1.append(node_arr_1)
            node_id1.append(node_index[i])
            origin_order1.append(np.array([node_id[i]]))
        if highway_one[i][1] != 0:
            node_arr_2 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_list2.append(node_arr_2)
            node_id2.append(node_index[i])
            origin_order2.append(np.array([node_id[i]]))
        if highway_one[i][2] != 0:
            node_arr_3 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_list3.append(node_arr_3)
            node_id3.append(node_index[i])
            origin_order3.append(np.array([node_id[i]]))
        if highway_one[i][3] != 0:
            node_arr_4 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_list4.append(node_arr_4)
            node_id4.append(node_index[i])
            origin_order4.append(np.array([node_id[i]]))
        if highway_one[i][4] != 0:
            node_arr_5 = np.array([[node_lon[i], node_lat[i], 0., 0., 0.], highway_one[i]])
            node_list5.append(node_arr_5)
            node_id5.append(node_index[i])
            origin_order5.append(np.array([node_id[i]]))

    edges = [edge_src, edge_dst]

    for i in range(len(edge_src)):
        if (edge_src[i] in node_id1) and (edge_dst[i] in node_id1):
            edge_src1.append(edge_src[i])
            edge_dst1.append(edge_dst[i])
        if (edge_src[i] in node_id2) and (edge_dst[i] in node_id2):
            edge_src2.append(edge_src[i])
            edge_dst2.append(edge_dst[i])
        if (edge_src[i] in node_id3) and (edge_dst[i] in node_id3):
            edge_src3.append(edge_src[i])
            edge_dst3.append(edge_dst[i])
        if (edge_src[i] in node_id4) and (edge_dst[i] in node_id4):
            edge_src4.append(edge_src[i])
            edge_dst4.append(edge_dst[i])
        if (edge_src[i] in node_id5) and (edge_dst[i] in node_id5):
            edge_src5.append(edge_src[i])
            edge_dst5.append(edge_dst[i])

    sort_dict1 = {}
    sort_dict2 = {}
    sort_dict3 = {}
    sort_dict4 = {}
    sort_dict5 = {}

    sort_dict = {}

    sort_dict1_reverse = {}
    sort_dict2_reverse = {}
    sort_dict3_reverse = {}
    sort_dict4_reverse = {}
    sort_dict5_reverse = {}

    for i in range(len(node_id)):
        sort_dict[node_id[i]] = i
        node_id[i] = i

    for i in range(len(node_id1)):
        sort_dict1[node_id1[i]] = i
        sort_dict1_reverse[i] = node_id1[i]
        node_id1[i] = i
    for i in range(len(node_id2)):
        sort_dict2[node_id2[i]] = i
        sort_dict2_reverse[i] = node_id2[i]
        node_id2[i] = i
    for i in range(len(node_id3)):
        sort_dict3[node_id3[i]] = i
        sort_dict3_reverse[i] = node_id3[i]
        node_id3[i] = i
    for i in range(len(node_id4)):
        sort_dict4[node_id4[i]] = i
        sort_dict4_reverse[i] = node_id4[i]
        node_id4[i] = i
    for i in range(len(node_id5)):
        sort_dict5[node_id5[i]] = i
        sort_dict5_reverse[i] = node_id5[i]
        node_id5[i] = i

    sort_dict_reverse_list = [sort_dict1_reverse, sort_dict2_reverse, sort_dict3_reverse, sort_dict4_reverse, sort_dict5_reverse]

    for i in range(len(edge_src1)):
        edge_src1[i] = sort_dict1[edge_src1[i]]
        edge_dst1[i] = sort_dict1[edge_dst1[i]]
    for i in range(len(edge_src2)):
        edge_src2[i] = sort_dict2[edge_src2[i]]
        edge_dst2[i] = sort_dict2[edge_dst2[i]]
    for i in range(len(edge_src3)):
        edge_src3[i] = sort_dict3[edge_src3[i]]
        edge_dst3[i] = sort_dict3[edge_dst3[i]]
    for i in range(len(edge_src4)):
        edge_src4[i] = sort_dict4[edge_src4[i]]
        edge_dst4[i] = sort_dict4[edge_dst4[i]]
    for i in range(len(edge_src5)):
        edge_src5[i] = sort_dict5[edge_src5[i]]
        edge_dst5[i] = sort_dict5[edge_dst5[i]]

    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    G4 = nx.Graph()
    G5 = nx.Graph()

    x1 = np.array(node_list1)
    x2 = np.array(node_list2)
    x3 = np.array(node_list3)
    x4 = np.array(node_list4)
    x5 = np.array(node_list5)

    G1.add_nodes_from(node_id1)
    G2.add_nodes_from(node_id2)
    G3.add_nodes_from(node_id3)
    G4.add_nodes_from(node_id4)
    G5.add_nodes_from(node_id5)

    edge_list1 = []
    edge_list2 = []
    edge_list3 = []
    edge_list4 = []
    edge_list5 = []

    for i in range(len(edge_src1)):
        edge_list1.append((edge_src1[i], edge_dst1[i]))
    for i in range(len(edge_src2)):
        edge_list2.append((edge_src2[i], edge_dst2[i]))
    for i in range(len(edge_src3)):
        edge_list3.append((edge_src3[i], edge_dst3[i]))
    for i in range(len(edge_src4)):
        edge_list4.append((edge_src4[i], edge_dst4[i]))
    for i in range(len(edge_src5)):
        edge_list5.append((edge_src5[i], edge_dst5[i]))

    G1.add_edges_from(edge_list1)
    G2.add_edges_from(edge_list2)
    G3.add_edges_from(edge_list3)
    G4.add_edges_from(edge_list4)
    G5.add_edges_from(edge_list5)

    x1 = np.array(node_list1)
    x2 = np.array(node_list2)
    x3 = np.array(node_list3)
    x4 = np.array(node_list4)
    x5 = np.array(node_list5)

    X = [x1, x2, x3, x4, x5]

    G = [G1, G2, G3, G4, G5]

    # print("G1 ", G1)

    node_id1 = np.array(node_id1)
    node_id2 = np.array(node_id2)
    node_id3 = np.array(node_id3)
    node_id4 = np.array(node_id4)
    node_id5 = np.array(node_id5)

    print("edge_1", len(edge_list1))
    print("edge_2", len(edge_list2))
    print("edge_3", len(edge_list3))
    print("edge_4", len(edge_list4))
    print("edge_5", len(edge_list5))

    origin_order = [node_id1, node_id2, node_id3, node_id4, node_id5]

    # print("origin_order", type(origin_order1))

    return G, X, edges, sort_dict_reverse_list, origin_order

# G, X = get_graph("Little Rock")
# print(G[0].nodes)



