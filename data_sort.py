import numpy as np
from data_encoding import one_hot_encoding
import matplotlib.pyplot as plt
import networkx as nx


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

def get_whole_graph(city_name):
    sort_dict, g, feature_list, P_ = one_hot_encoding(city_name)

    feats_1, adj_1, prior_order1, s1 = get_graph_data(sort_dict, g[0], feature_list[0], 1)
    feats_2, adj_2, prior_order2, s2 = get_graph_data(sort_dict, g[1], feature_list[1], 2, P_, prior_order1)
    feats_3, adj_3, prior_order3, s3 = get_graph_data(sort_dict, g[2], feature_list[2], 3, P_, prior_order2)
    feats_4, adj_4, prior_order4, s4 = get_graph_data(sort_dict, g[3], feature_list[3], 4, P_, prior_order3)
    feats_5, adj_5, prior_order5, s5 = get_graph_data(sort_dict, g[4], feature_list[4], 5, P_, prior_order4)

    pos_dict = {}

    node_list = []
    feature_list = []

    node_list1 = []
    node_list2 = []
    node_list3 = []
    node_list4 = []
    node_list5 = []

    final_g = nx.Graph()

    for i in range(len(prior_order1)):
        node_list.append(s1[0][prior_order1[i]])
        feature_list.append(feats_1[i])

    node_list1 = node_list

    for i in range(len(prior_order2)):
        if s2[1][prior_order2[i]] not in node_list:
            node_list.append(s2[1][prior_order2[i]])
            feature_list.append(feats_2[i])

    node_list2 = node_list

    for i in range(len(prior_order3)):
        if s3[2][prior_order3[i]] not in node_list:
            node_list.append(s3[2][prior_order3[i]])
            feature_list.append(feats_3[i])
    print("3 node", len(node_list))

    node_list3 = node_list

    for i in range(len(prior_order4)):
        if s3[3][prior_order4[i]] not in node_list:
            node_list.append(s3[3][prior_order4[i]])
            feature_list.append(feats_4[i])

    node_list4 = node_list

    for i in range(len(prior_order5)):
        if s4[4][prior_order5[i]] not in node_list:
            node_list.append(s3[4][prior_order5[i]])
            feature_list.append(feats_5[i])

    print(feature_list)


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

    return final_g, adj_, feature_list


