import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

def one_hot_encoding(city_name,base_dir,level):
    city_name_file = (city_name.lower()).replace(" ", "")
    edge_norm_file_name = base_dir + str(level) + "/" + city_name_file + "_edge_norm_ver1_highway.csv"
    node_norm_file_name = base_dir + str(level) + "/" + city_name_file + "_node_norm.csv"
    node_highway_file_name = base_dir + str(level) + "/" + city_name_file + "_node_norm_highway.csv"

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

    highway_one = []

    highway1 = 0



    for i in range(len(highway_dict)):
        for j in range(len(highway_dict[node_index[i]])):
            if highway_dict[node_index[i]][j] in highway_list1:
                highway1 = 1
        highway_one.append([highway1])
        highway1 = 0


    feature_list1 = []
    node_id1 = []




    for i in range(len(node_index)):
        if highway_one[i][0] != 0:
            feature_arr_1 = np.array([node_lat[i], node_lon[i]])
            feature_list1.append(feature_arr_1)
            node_id1.append(node_index[i])


    sort_dict1 = {}


    edge_src1 = []
    edge_dst1 = []





    P1_2 = []
    P1_3 = []
    P1_4 = []






    for i in range(len(edge_src)):
        if (edge_src[i] in node_id1) and (edge_dst[i] in node_id1):
            edge_src1.append(edge_src[i])
            edge_dst1.append(edge_dst[i])



    node_sort_id1 = node_id1

    for i in range(len(node_id1)):
        sort_dict1[node_id1[i]] = i
        node_sort_id1[i] = i





    edge_src_sort1 = edge_src1
    edge_dst_sort1 = edge_dst1


    for i in range(len(edge_src1)):
        edge_src_sort1[i] = sort_dict1[edge_src1[i]]
        edge_dst_sort1[i] = sort_dict1[edge_dst1[i]]




    g1 = nx.Graph()

    g1.add_nodes_from(node_sort_id1)



    for i in range(len(edge_src_sort1)):
        g1.add_edge(edge_src_sort1[i], edge_dst_sort1[i])



    feature_list1 = np.array(feature_list1)


    pos_dict = {}


    """
    for i in range(len(feature_list2)):
        pos_dict[list(g2.nodes)[i]] = (feature_list2[i][0], feature_list2[i][1])

    pos = pos_dict
    nx.draw_networkx_nodes(g2, pos, node_size=3, node_color='black')
    nx.draw_networkx_edges(g2, pos, alpha=0.5, width=1)
    plt.axis('off')
    plt.show()
    save_path = "./datasets/cities/" + "Firenze" + ".png"
    plt.savefig(save_path)

    """
    sort_dict = [sort_dict1]
    g = [g1]
    feature_list = [feature_list1]
    P_ = [P1_2, P1_3, P1_4]


    return sort_dict, g, feature_list, P_
