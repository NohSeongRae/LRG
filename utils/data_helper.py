###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import os
import torch
import pickle
import json

from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import sparse as sp
import networkx as nx
import torch.nn.functional as F
import pygsp
import pandas as pd
__all__ = [
    "save_graph_list",
    "load_graph_list",
    "graph_load_batch",
    "preprocess_graph_list",
    "create_graphs",
]
from utils.level4.data_sort import get_whole_graph_4
from utils.level1.data_sort import get_whole_graph_1

def one_hot_encoding(city_name,base_dir,level):
    city_name_file = (city_name.lower()).replace(" ", "")
    edge_norm_file_name = base_dir +str(level)+"/"+ city_name_file + "_edge_norm_ver1_highway.csv"
    node_norm_file_name = base_dir+str(level)+"/" + city_name_file + "_node_norm.csv"
    node_highway_file_name = base_dir +str(level)+"/"+ city_name_file + "_node_norm_highway.csv"

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
    highway_list2 = ["secondary", "secondary_link"]
    highway_list3 = ["tertiary", "tertiary_link"]
    highway_list4 = ["unclassified"]
    highway_list5 = ["residential"]

    highway_one = []

    highway1 = 0
    highway2 = 0
    highway3 = 0
    highway4 = 0
    highway5 = 0

    for i in range(len(highway_dict)):
        for j in range(len(highway_dict[node_index[i]])):
            if highway_dict[node_index[i]][j] in highway_list1:
                highway1 = 1
            if highway_dict[node_index[i]][j] in highway_list2:
                highway2 = 1
            if highway_dict[node_index[i]][j] in highway_list3:
                highway3 = 1
            if highway_dict[node_index[i]][j] in highway_list4:
                highway4 = 1
            if highway_dict[node_index[i]][j] in highway_list5:
                highway5 = 1
        highway_one.append([highway1, highway2, highway3, highway4, highway5])
        highway1 = 0
        highway2 = 0
        highway3 = 0
        highway4 = 0
        highway5 = 0

    feature_list1 = []
    node_id1 = []

    feature_list2 = []
    node_id2 = []

    feature_list3 = []
    node_id3 = []

    feature_list4 = []
    node_id4 = []

    feature_list5 = []
    node_id5 = []

    for i in range(len(node_index)):
        if highway_one[i][0] != 0:
            feature_arr_1 = np.array([node_lon[i], node_lat[i]])
            feature_list1.append(feature_arr_1)
            node_id1.append(node_index[i])
        if highway_one[i][1] != 0:
            feature_arr_2 = np.array([node_lon[i], node_lat[i]])
            feature_list2.append(feature_arr_2)
            node_id2.append(node_index[i])
        if highway_one[i][2] != 0:
            feature_arr_3 = np.array([node_lon[i], node_lat[i]])
            feature_list3.append(feature_arr_3)
            node_id3.append(node_index[i])
        if highway_one[i][3] != 0:
            feature_arr_4 = np.array([node_lon[i], node_lat[i]])
            feature_list4.append(feature_arr_4)
            node_id4.append(node_index[i])
        if highway_one[i][4] != 0:
            feature_arr_5 = np.array([node_lon[i], node_lat[i]])
            feature_list5.append(feature_arr_5)
            node_id5.append(node_index[i])


    sort_dict1 = {}
    sort_dict2 = {}
    sort_dict3 = {}
    sort_dict4 = {}
    sort_dict5 = {}

    edge_src1 = []
    edge_dst1 = []

    edge_src2 = []
    edge_dst2 = []

    edge_src3 = []
    edge_dst3 = []

    edge_src4 = []
    edge_dst4 = []

    edge_src5 = []
    edge_dst5 = []

    P1_2 = []
    P1_3 = []
    P1_4 = []
    P1_5 = []

    P2_3 = []
    P2_4 = []
    P2_5 = []

    P3_4 = []
    P3_5 = []

    P4_5 = []

    for i in range(len(node_id1)):
        if node_id1[i] in node_id2:
            P1_2.append(node_id1[i])
        if node_id1[i] in node_id3:
            P1_3.append(node_id1[i])
        if node_id1[i] in node_id4:
            P1_4.append(node_id1[i])
        if node_id1[i] in node_id5:
            P1_5.append(node_id1[i])

    for i in range(len(node_id2)):
        if node_id2[i] in node_id3:
            P2_3.append(node_id2[i])
        if node_id2[i] in node_id4:
            P2_4.append(node_id2[i])
        if node_id2[i] in node_id5:
            P2_5.append(node_id2[i])

    for i in range(len(node_id3)):
        if node_id3[i] in node_id4:
            P3_4.append(node_id3[i])
        if node_id3[i] in node_id5:
            P3_5.append(node_id3[i])

    for i in range(len(node_id4)):
        if node_id4[i] in node_id5:
            P4_5.append(node_id4[i])

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

    node_sort_id1 = node_id1
    node_sort_id2 = node_id2
    node_sort_id3 = node_id3
    node_sort_id4 = node_id4
    node_sort_id5 = node_id5

    for i in range(len(node_id1)):
        sort_dict1[node_id1[i]] = i
        node_sort_id1[i] = i

    for i in range(len(node_id2)):
        sort_dict2[node_id2[i]] = i
        node_sort_id2[i] = i

    for i in range(len(node_id3)):
        sort_dict3[node_id3[i]] = i
        node_sort_id3[i] = i

    for i in range(len(node_id4)):
        sort_dict4[node_id4[i]] = i
        node_sort_id4[i] = i

    for i in range(len(node_id5)):
        sort_dict5[node_id5[i]] = i
        node_sort_id5[i] = i


    edge_src_sort1 = edge_src1
    edge_dst_sort1 = edge_dst1

    edge_src_sort2 = edge_src2
    edge_dst_sort2 = edge_dst2

    edge_src_sort3 = edge_src3
    edge_dst_sort3 = edge_dst3

    edge_src_sort4 = edge_src4
    edge_dst_sort4 = edge_dst4

    edge_src_sort5 = edge_src5
    edge_dst_sort5 = edge_dst5

    for i in range(len(edge_src1)):
        edge_src_sort1[i] = sort_dict1[edge_src1[i]]
        edge_dst_sort1[i] = sort_dict1[edge_dst1[i]]

    for i in range(len(edge_src2)):
        edge_src_sort2[i] = sort_dict2[edge_src2[i]]
        edge_dst_sort2[i] = sort_dict2[edge_dst2[i]]

    for i in range(len(edge_src3)):
        edge_src_sort3[i] = sort_dict3[edge_src3[i]]
        edge_dst_sort3[i] = sort_dict3[edge_dst3[i]]

    for i in range(len(edge_src4)):
        edge_src_sort4[i] = sort_dict4[edge_src4[i]]
        edge_dst_sort4[i] = sort_dict4[edge_dst4[i]]

    for i in range(len(edge_src5)):
        edge_src_sort5[i] = sort_dict5[edge_src5[i]]
        edge_dst_sort5[i] = sort_dict5[edge_dst5[i]]

    g1 = nx.Graph()
    g2 = nx.Graph()
    g3 = nx.Graph()
    g4 = nx.Graph()
    g5 = nx.Graph()

    g1.add_nodes_from(node_sort_id1)
    g2.add_nodes_from(node_sort_id2)
    g3.add_nodes_from(node_sort_id3)
    g4.add_nodes_from(node_sort_id4)
    g5.add_nodes_from(node_sort_id5)

    for i in range(len(edge_src_sort1)):
        g1.add_edge(edge_src_sort1[i], edge_dst_sort1[i])

    for i in range(len(edge_src_sort2)):
        g2.add_edge(edge_src_sort2[i], edge_dst_sort2[i])

    for i in range(len(edge_src_sort3)):
        g3.add_edge(edge_src_sort3[i], edge_dst_sort3[i])

    for i in range(len(edge_src_sort4)):
        g4.add_edge(edge_src_sort4[i], edge_dst_sort4[i])

    for i in range(len(edge_src_sort5)):
        g5.add_edge(edge_src_sort5[i], edge_dst_sort5[i])

    feature_list1 = np.array(feature_list1)
    feature_list2 = np.array(feature_list2)
    feature_list3 = np.array(feature_list3)
    feature_list4 = np.array(feature_list4)
    feature_list5 = np.array(feature_list5)

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
    sort_dict = [sort_dict1, sort_dict2, sort_dict3, sort_dict4, sort_dict5]
    g = [g1, g2, g3, g4, g5]
    feature_list = [feature_list1, feature_list2, feature_list3, feature_list4, feature_list5]
    P_ = [P1_2, P1_3, P1_4, P1_5, P2_3, P2_4, P2_5, P3_4, P3_5, P4_5]

    # print(feature_list[3])

    return sort_dict, g, feature_list, P_
def get_graph_data(sort_dict, g, feature_list, num, p_=None, prior_node="None"):
    node_degree_list = [(n, d) for n, d in g.degree()]

    ### BFS & DFS from largest-degree node
    CGs = [g.subgraph(c) for c in nx.connected_components(g)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    sort_dict1_reverse = {v: k for k, v in sort_dict[0].items()}
    sort_dict2_reverse = {v: k for k, v in sort_dict[1].items()}
    sort_dict3_reverse = {v: k for k, v in sort_dict[2].items()}
    sort_dict4_reverse = {v: k for k, v in sort_dict[3].items()}
    sort_dict5_reverse = {v: k for k, v in sort_dict[4].items()}

    node_list_bfs = []
    node_list_dfs = []

    start_node = "None"

    in_node = []
    index_list = []

    print("CGs", len(CGs))

    for ii in range(len(CGs)):
        if num == 2:
            for j in range(len(CGs[ii].nodes)):
                if sort_dict2_reverse[list(CGs[ii].nodes)[j]] in p_[0]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                start_node = "None"
            index_list.clear()
            in_node.clear()

        node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
        degree_sequence = sorted(
            node_degree_list, key=lambda tt: tt[1], reverse=True
        )

        if num == 3:
            for j in range(len(CGs[ii].nodes)):
                if sort_dict3_reverse[list(CGs[ii].nodes)[j] in p_[4]]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                if sort_dict3_reverse[list(CGs[ii].nodes)[j] in p_[1]]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                start_node = "None"
            index_list.clear()
            in_node.clear()

        if num == 4:
            for j in range(len(CGs[ii].nodes)):
                if sort_dict4_reverse[list(CGs[ii].nodes)[j] in p_[7]]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                if sort_dict4_reverse[list(CGs[ii].nodes)[j] in p_[5]]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                if sort_dict4_reverse[list(CGs[ii].nodes)[j] in p_[2]]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                start_node = "None"
            index_list.clear()
            in_node.clear()

        if num == 5:
            for j in range(len(CGs[ii].nodes)):
                if sort_dict5_reverse[list(CGs[ii].nodes)[j] in p_[9]]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                if sort_dict5_reverse[list(CGs[ii].nodes)[j] in p_[8]]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                if sort_dict5_reverse[list(CGs[ii].nodes)[j] in p_[6]]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                if sort_dict5_reverse[list(CGs[ii].nodes)[j] in p_[3]]:
                    in_node.append(list(CGs[ii].nodes)[j])
            if len(in_node) > 0:
                for i in range(len(in_node)):
                    if in_node[i] in prior_node:
                        index_list.append(prior_node.index(in_node[i]))
                if len(index_list) > 0:
                    start_node = prior_node[min(index_list)]
                else:
                    start_node = "None"
            else:
                start_node = "None"
            index_list.clear()
            in_node.clear()

        if start_node == "None":
            start_node = degree_sequence[0][0]

        bfs_tree = nx.bfs_tree(CGs[ii], source=start_node)
        dfs_tree = nx.dfs_tree(CGs[ii], source=start_node)

        node_list_bfs += list(bfs_tree.nodes())
        node_list_dfs += list(dfs_tree.nodes())

    prior_node = node_list_dfs
    # print(prior_node)

    feats_1 = feature_list[node_list_dfs]
    adj_1 = nx.to_numpy_array(g, nodelist=node_list_dfs)

    sort_dict_reverse = [sort_dict1_reverse, sort_dict2_reverse, sort_dict3_reverse, sort_dict4_reverse,
                         sort_dict5_reverse]

    return feats_1, adj_1, prior_node, sort_dict_reverse

def get_whole_graph(city_name,base_dir, level):
    sort_dict, g, feature_list, P_ = one_hot_encoding(city_name,base_dir, level)

    feats_1, adj_1, prior_order1, s1 = get_graph_data(sort_dict, g[0], feature_list[0], 1)
    feats_2, adj_2, prior_order2, s2 = get_graph_data(sort_dict, g[1], feature_list[1], 2, P_, prior_order1)
    feats_3, adj_3, prior_order3, s3 = get_graph_data(sort_dict, g[2], feature_list[2], 3, P_, prior_order2)
    feats_4, adj_4, prior_order4, s4 = get_graph_data(sort_dict, g[3], feature_list[3], 4, P_, prior_order3)
    feats_5, adj_5, prior_order5, s5 = get_graph_data(sort_dict, g[4], feature_list[4], 5, P_, prior_order4)

    pos_dict = {}

    node_list = []
    feature_lists = []

    node_list1 = []
    node_list2 = []
    node_list3 = []
    node_list4 = []
    node_list5 = []

    final_g = nx.Graph()

    for i in range(len(prior_order1)):
        node_list.append(s1[0][prior_order1[i]])
        feature_lists.append(feats_1[i])

    node_list1 = node_list

    for i in range(len(prior_order2)):
        if s2[1][prior_order2[i]] not in node_list:
            node_list.append(s2[1][prior_order2[i]])
            feature_lists.append(feats_2[i])

    node_list2 = node_list

    for i in range(len(prior_order3)):
        if s3[2][prior_order3[i]] not in node_list:
            node_list.append(s3[2][prior_order3[i]])
            feature_lists.append(feats_3[i])
    # print("3 node", len(node_list))

    node_list3 = node_list

    for i in range(len(prior_order4)):
        if s3[3][prior_order4[i]] not in node_list:
            node_list.append(s3[3][prior_order4[i]])
            feature_lists.append(feats_4[i])

    node_list4 = node_list

    for i in range(len(prior_order5)):
        if s4[4][prior_order5[i]] not in node_list:
            node_list.append(s3[4][prior_order5[i]])
            feature_lists.append(feats_5[i])


    final_g.add_nodes_from(node_list)


    for i in range(len(g[0].edges)):
        final_g.add_edge(s1[0][list(g[0].edges)[i][0]], s1[0][list(g[0].edges)[i][1]])

    for i in range(len(g[1].edges)):
        final_g.add_edge(s1[1][list(g[1].edges)[i][0]], s1[1][list(g[1].edges)[i][1]])

    for i in range(len(g[2].edges)):
        final_g.add_edge(s1[2][list(g[2].edges)[i][0]], s1[2][list(g[2].edges)[i][1]])

    for i in range(len(g[3].edges)):
        final_g.add_edge(s1[3][list(g[3].edges)[i][0]], s1[3][list(g[3].edges)[i][1]])

    for i in range(len(g[4].edges)):
        final_g.add_edge(s1[4][list(g[4].edges)[i][0]], s1[4][list(g[4].edges)[i][1]])

    adj_ = nx.to_numpy_array(final_g)
    final_g=[final_g]
    feature_lists = [np.array(feature_lists)]
    # print(f"node_feats_list: {feature_lists}")
    return final_g, adj_, feature_lists


# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def pick_connected_component_new(G):
    # import pdb; pdb.set_trace()

    # adj_list = G.adjacency_list()
    # for id,adj in enumerate(adj_list):
    #     id_min = min(adj)
    #     if id<id_min and id>=1:
    #     # if id<id_min and id>=4:
    #         break
    # node_list = list(range(id)) # only include node prior than node "id"

    adj_dict = nx.to_dict_of_lists(G)
    for node_id in sorted(adj_dict.keys()):
        id_min = min(adj_dict[node_id])
        if node_id < id_min and node_id >= 1:
            # if node_id<id_min and node_id>=4:
            break
    node_list = list(range(node_id))  # only include node prior than node "node_id"

    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G


def load_graph_list(fname, is_real=True):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)

    # import pdb; pdb.set_trace()
    for i in range(len(graph_list)):
        edges_with_selfloops = list(graph_list[i].selfloop_edges())
        if len(edges_with_selfloops) > 0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            graph_list[i] = max(
                nx.connected_component_subgraphs(graph_list[i]), key=len
            )
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])
    return graph_list


def preprocess_graph_list(graph_list):
    for i in range(len(graph_list)):
        edges_with_selfloops = list(graph_list[i].selfloop_edges())
        if len(edges_with_selfloops) > 0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            graph_list[i] = max(
                nx.connected_component_subgraphs(graph_list[i]), key=len
            )
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])
    return graph_list


def graph_load_batch(
    data_dir,
    min_num_nodes=20,
    max_num_nodes=1000,
    name="ENZYMES",
    node_attributes=True,
    graph_labels=True,
):
    """
    load many graphs, e.g. enzymes
    :return: a list of graphs
    """
    print("Loading graph dataset: " + str(name))
    G = nx.Graph()
    # load data
    path = os.path.join(data_dir, name)
    data_adj = np.loadtxt(
        os.path.join(path, "{}_A.txt".format(name)), delimiter=","
    ).astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(
            os.path.join(path, "{}_node_attributes.txt".format(name)), delimiter=","
        )
    data_node_label = np.loadtxt(
        os.path.join(path, "{}_node_labels.txt".format(name)), delimiter=","
    ).astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, "{}_graph_indicator.txt".format(name)), delimiter=","
    ).astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            os.path.join(path, "{}_graph_labels.txt".format(name)), delimiter=","
        ).astype(int)

    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph["label"] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if (
            G_sub.number_of_nodes() >= min_num_nodes
            and G_sub.number_of_nodes() <= max_num_nodes
        ):
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
            # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
            # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print("Loaded")
    return graphs


def create_graphs(graph_type,level,base_dir, data_dir="data", noise=10.0, seed=1234):
    ### load datasets
    graphs = []
    feats = []
    # synthetic graphs

    if graph_type == "gecko":

        with open(os.path.join(data_dir, "layer2_65_feats_std.pickle"), "rb") as handle:
            data = pickle.load(handle)
            # data = data[:100]
        for elem in data:
            g = nx.Graph()
            g.add_nodes_from(elem["nodes"])
            g.add_edges_from(elem["edges"])
            g.remove_edges_from(nx.selfloop_edges(g))
            feats_ = elem["feats"]
            feats.append(feats_)
            graphs.append(g)

    if graph_type=="grid":
        graphs=[]
        feats=[]
        for i in range(10,20):
            for j in range(10,20):
                graphs.append(nx.grid_2d_graph(i,j))
                g_temp = pygsp.graphs.Grid2d(N1=i, N2=j)
                feats_=g_temp.coords.astype(np.float32)
                feats.append(feats_)
        #         print(f"feats_: {feats_}")
        #
        # print(f"feats: {feats}")

    elif graph_type == 'Firenze':
        if level==5:
            graphs, adj_, feats = get_whole_graph("Firenze",base_dir, level)
        elif level==4:
            graphs, adj_, feats =get_whole_graph_4("Firenze",base_dir, level)
        elif level==1:
            graphs, adj_, feats=get_whole_graph_1("Firenze",base_dir, level)





    elif graph_type == 'FIRSTMM_DB':
        graphs = graph_load_batch(
            data_dir,
            min_num_nodes=0,
            max_num_nodes=10000,
            name='FIRSTMM_DB',
            node_attributes=False,
            graph_labels=True)

    num_nodes = [gg.number_of_nodes() for gg in graphs]
    num_edges = [gg.number_of_edges() for gg in graphs]
    print(
        "max # nodes = {} || mean # nodes = {}".format(
            max(num_nodes), np.mean(num_nodes)
        )
    )
    print(
        "max # edges = {} || mean # edges = {}".format(
            max(num_edges), np.mean(num_edges)
        )
    )

    if len(feats) > 0:
        return graphs, feats
    else:
        feats=[]
        return graphs, feats

