import networkx as nx

import numpy as np

from collections import defaultdict
from data_one_hot_encode import get_graph2


"""
아래 수정필요 
"""
num_canonical_order = 1

def get_graph_data(g, node_order, node_feats):
    # print("type", type(node_feats))
    node_degree_list = [(n, d) for n, d in g.degree()]
    # print(len(g.nodes))
    # print(node_degree_list)

    adj_0 = np.array(nx.to_numpy_array(g))

    if node_feats is not None:
        feats_0 = node_feats

    ### Degree descent ranking
    degree_sequence = sorted(node_degree_list,  key=lambda tt: tt[1], reverse=True)
    adj_1 = np.array(
        nx.to_numpy_array(g, nodelist=[dd[0] for dd in degree_sequence])
    )

    if node_feats is not None:
        feats_1 = node_feats[[dd[0] for dd in degree_sequence]]

    ### Degree ascent ranking
    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
    adj_2 = np.array(
        nx.to_numpy_array(g, nodelist=[dd[0] for dd in degree_sequence])
    )

    if node_feats is not None:
        feats_2 = node_feats[[dd[0] for dd in degree_sequence]]

    ### BFS & DFS from largest-degree node
    CGs = [g.subgraph(c) for c in nx.connected_components(g)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    node_list_bfs = []
    node_list_dfs = []
    for ii in range(len(CGs)):
        node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
        degree_sequence = sorted(
            node_degree_list, key=lambda tt: tt[1], reverse=True
        )

        bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
        dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

        node_list_bfs += list(bfs_tree.nodes())
        node_list_dfs += list(dfs_tree.nodes())

    adj_3 = np.array(nx.to_numpy_array(g, nodelist=node_list_bfs))
    adj_4 = np.array(nx.to_numpy_array(g, nodelist=node_list_dfs))

    if node_feats is not None:
        feats_3 = node_feats[node_list_bfs]
        feats_4 = node_feats[node_list_dfs]

    ### k-core
    num_core = nx.core_number(g)
    core_order_list = sorted(list(set(num_core.values())), reverse=True)
    degree_dict = dict(g.degree())
    core_to_node = defaultdict(list)
    for nn, kk in num_core.items():
        core_to_node[kk] += [nn]

    node_list = []
    for kk in core_order_list:
        sort_node_tuple = sorted(
            [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
            key=lambda tt: tt[1],
            reverse=True,
        )
        node_list += [nn for nn, dd in sort_node_tuple]

    adj_5 = np.array(nx.to_numpy_array(g, nodelist=node_list))
    if node_feats is not None:
        feats_5 = node_feats[node_list]

    if num_canonical_order == 5:
        adj_list = [adj_0, adj_1, adj_3, adj_4, adj_5]

        if node_feats is not None:
            node_feats_list = [feats_0, feats_1, feats_2, feats_3, feats_4, feats_5]
    else:
        if node_order == "degree_decent":
            adj_list = [adj_1]
            if node_feats is not None:
                node_feats_list = [feats_1]
        elif node_order == "degree_accent":
            adj_list = [adj_2]
            if node_feats is not None:
                node_feats_list = [feats_2]
        elif node_order == "BFS":
            adj_list = [adj_3]
            if node_feats is not None:
                node_feats_list = [feats_3]
        elif node_order == "DFS":
            adj_list = [adj_4]
            if node_feats is not None:
                node_feats_list = [feats_4]
        elif node_order == "k_core":
            adj_list = [adj_5]
            if node_feats is not None:
                node_feats_list = [feats_5]
        elif node_order == "DFS+BFS":
            adj_list = [adj_4, adj_3]
            if node_feats is not None:
                node_feats_list = [feats_4, feats_5]
        elif node_order == "DFS+BFS+k_core":
            adj_list = [adj_4, adj_3, adj_5]
            if node_feats is not None:
                node_feats_list = [feats_4, feats_3, feats_5]
        elif node_order == "DFS+BFS+k_core+degree_decent":
            adj_list = [adj_4, adj_3, adj_5, adj_1]
            if node_feats is not None:
                node_feats_list = [feats_4, feats_3, feats_5, feats_1]
        elif node_order == "all":
            adj_list = [adj_4, adj_3, adj_5, adj_1, adj_0]
            if node_feats is not None:
                node_feats_list = [feats_4, feats_3, feats_5, feats_1, feats_0]
        else:
            adj_list = [adj_0]
            if node_feats is not None:
                node_feats_list = [feats_0]

    return adj_list, node_feats_list

# city_name = "Little Rock"

def get_whole_graph(city_name, node_order):

    G, A, X, node_ids = get_graph2(city_name)
    A_, node_feats = get_graph_data(G, node_order, X)
    G_ = nx.from_numpy_array(A_[0])

    return G_, A_, node_feats

G_, A_, node_feats = get_whole_graph("Firenze", "BFS")

