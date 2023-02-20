import torch
import time
import os
import pickle
import glob
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from utils.data_helper import *


class GRANData(object):
    def __init__(self, config, graphs, node_feats=None, tag="train"):
        self.config = config
        self.data_path = config.dataset.data_path
        self.model_name = config.model.name
        self.max_num_nodes = config.model.max_num_nodes
        self.block_size = config.model.block_size
        self.stride = config.model.sample_stride

        self.graphs = graphs
        self.has_nodes_feats=config.dataset.has_node_feat
        self.node_feats = node_feats
        self.num_graphs = len(graphs)
        self.npr = np.random.RandomState(config.seed)
        self.node_order = config.dataset.node_order
        self.num_canonical_order = config.model.num_canonical_order
        self.tag = tag
        self.num_fwd_pass = config.dataset.num_fwd_pass
        self.is_sample_subgraph = config.dataset.is_sample_subgraph
        self.num_subgraph_batch = config.dataset.num_subgraph_batch
        self.is_overwrite_precompute = config.dataset.is_overwrite_precompute
        self.city_level=config.dataset.city_level
        if config.dataset.name == "Firenze":
            self.is_firenze=True
        else:
            self.is_firenze=False

        if self.is_sample_subgraph:
            assert self.num_subgraph_batch > 0

        self.save_path = os.path.join(
            self.data_path,
            "{}_{}_{}_{}_{}_{}_{}_{}_precompute_small".format(
                config.model.name,
                config.dataset.name,
                tag,
                self.block_size,
                self.stride,
                self.num_canonical_order,
                self.node_order,
                self.city_level
            ),
        )

        if not os.path.isdir(self.save_path) or self.is_overwrite_precompute:
            self.file_names = []
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

            # print(f"self.node_feats: {self.node_feats}")
            self.config.dataset.save_path = self.save_path
            for index in tqdm(range(self.num_graphs)):
                G = self.graphs[index]
                if self.is_firenze == True:
                    adj_0 = np.array(nx.to_numpy_array(G))
                    data=([adj_0], [self.node_feats])
                else:
                    if self.has_nodes_feats:
                        data = self._get_graph_data(G, node_feats=self.node_feats[index])
                    else:
                        data = self._get_graph_data(G)
                tmp_path = os.path.join(self.save_path, "{}_{}.p".format(tag, index))
                pickle.dump(data, open(tmp_path, "wb"))
                self.file_names += [tmp_path]
        else:
            self.file_names = glob.glob(os.path.join(self.save_path, "*.p"))

    def _get_graph_data(self, G, node_feats=None):
        node_degree_list = [(n, d) for n, d in G.degree()] #((4,7),d)

        adj_0 = np.array(nx.to_numpy_array(G))

        if node_feats is not None:
            feats_0 = node_feats

        # print(f"_get_graph_data, node_feats: {node_feats}")

        ### Degree descent ranking
        # N.B.: largest-degree node may not be unique
        degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
        adj_1 = np.array(
            nx.to_numpy_array(G, nodelist=[dd[0] for dd in degree_sequence])
        )

        if node_feats is not None:
            feats_1 = node_feats[[dd[0] for dd in degree_sequence]]

        ### Degree ascent ranking
        degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
        adj_2 = np.array(
            nx.to_numpy_array(G, nodelist=[dd[0] for dd in degree_sequence])
        )
        if node_feats is not None:
            feats_2 = node_feats[[dd[0] for dd in degree_sequence]]

        ### BFS & DFS from largest-degree node
        CGs = [G.subgraph(c) for c in nx.connected_components(G)]


        # rank connected componets from large to small size
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

        node_list_bfs = []
        node_list_dfs = []
        # count=1 #debug

        for ii in range(len(CGs)):
            node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
            degree_sequence = sorted(
                node_degree_list, key=lambda tt: tt[1], reverse=True
            )

            bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
            dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

            node_list_bfs += list(bfs_tree.nodes())
            node_list_dfs += list(dfs_tree.nodes())
            # print(f"dfs_tree.nodes(): {dfs_tree.nodes()}")
            # print(f"node_degree_list: {node_degree_list}")


        adj_3 = np.array(nx.to_numpy_array(G, nodelist=node_list_bfs))
        # print(f"adj_3.shape: {adj_3.shape}")
        adj_4 = np.array(nx.to_numpy_array(G, nodelist=node_list_dfs))

        # print(f"node_feats _ before: {node_feats}")
        if node_feats is not None:
            node_list_dfs_s=np.array(node_list_dfs)
            feats_3 = node_feats[node_list_bfs]
            feats_4 = node_feats[node_list_dfs_s[:,0]] # 이 코드가 문제임
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            # if count==1:
            #     # print(f"node_feats _ after: {feats_4}")
            #     # print(f" dfs, node_list_dfs_s: {node_list_dfs_s}")
            #     count+=1


        ### k-core
        num_core = nx.core_number(G)
        core_order_list = sorted(list(set(num_core.values())), reverse=True)
        degree_dict = dict(G.degree())
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

        adj_5 = np.array(nx.to_numpy_array(G, nodelist=node_list))
        if node_feats is not None:
            feats_5 = node_feats[node_list]

        if self.num_canonical_order == 5:
            adj_list = [adj_0, adj_1, adj_3, adj_4, adj_5]

            if self.node_feats is not None:
                node_feats_list = [feats_0, feats_1, feats_2, feats_3, feats_4, feats_5]
        else:
            if self.node_order == "degree_decent":
                adj_list = [adj_1]
                if self.node_feats is not None:
                    node_feats_list = [feats_1]
            elif self.node_order == "degree_accent":
                adj_list = [adj_2]
                if self.node_feats is not None:
                    node_feats_list = [feats_2]
            elif self.node_order == "BFS":
                adj_list = [adj_3]
                if self.node_feats is not None:
                    node_feats_list = [feats_3]
            elif self.node_order == "DFS":
                adj_list = [adj_4]
                if self.has_nodes_feats is False:
                    node_feats_list=[]
                if (self.node_feats is not None) and (self.has_nodes_feats == True) :
                    node_feats_list = [feats_4]
            elif self.node_order == "k_core":
                adj_list = [adj_5]
                if self.node_feats is not None:
                    node_feats_list = [feats_5]
            elif self.node_order == "DFS+BFS":
                adj_list = [adj_4, adj_3]
                if self.node_feats is not None:
                    node_feats_list = [feats_4, feats_5]
            elif self.node_order == "DFS+BFS+k_core":
                adj_list = [adj_4, adj_3, adj_5]
                if self.node_feats is not None:
                    node_feats_list = [feats_4, feats_3, feats_5]
            elif self.node_order == "DFS+BFS+k_core+degree_decent":
                adj_list = [adj_4, adj_3, adj_5, adj_1]
                if self.node_feats is not None:
                    node_feats_list = [feats_4, feats_3, feats_5, feats_1]
            elif self.node_order == "all":
                adj_list = [adj_4, adj_3, adj_5, adj_1, adj_0]
                if self.node_feats is not None:
                    node_feats_list = [feats_4, feats_3, feats_5, feats_1, feats_0]
            else:
                adj_list = [adj_0]
                if self.node_feats is not None:
                    node_feats_list = [feats_0]

        # print('number of nodes = {}'.format(adj_0.shape[0]))
        # print(f"_get_graph_data, node_feats_list: {node_feats_list}")
        if self.node_feats is not None:
            return adj_list, node_feats_list
        else:
            return adj_list

    def __getitem__(self, index):
        K = self.block_size
        N = self.max_num_nodes
        S = self.stride

        # load graph
        tmp = pickle.load(open(self.file_names[index], "rb"))
        if type(tmp) == tuple:
            adj_list, node_feats_list = tmp

        else:
            (adj_list,) = (tmp,)
        num_nodes = adj_list[0].shape[0]

        # Number of subgraph: (n_rows-block_size)/stride + 1
        num_subgraphs = int(np.floor((num_nodes - K) / S) + 1)

        if self.is_sample_subgraph:
            if self.num_subgraph_batch < num_subgraphs:
                num_subgraphs_pass = int(
                    np.floor(self.num_subgraph_batch / self.num_fwd_pass)
                )
            else:
                num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))

            end_idx = min(num_subgraphs, self.num_subgraph_batch)
        else:
            num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))
            end_idx = num_subgraphs

        ### random permute subgraphs
        rand_perm_idx = self.npr.permutation(num_subgraphs).tolist()

        start_time = time.time()
        data_batch = []
        for ff in range(self.num_fwd_pass):
            ff_idx_start = num_subgraphs_pass * ff
            if ff == self.num_fwd_pass - 1:
                ff_idx_end = end_idx
            else:
                ff_idx_end = (ff + 1) * num_subgraphs_pass

            rand_idx = rand_perm_idx[ff_idx_start:ff_idx_end]

            edges = []
            node_idx_gnn = []
            node_idx_feat = []
            label = []
            subgraph_size = []
            subgraph_idx = []
            att_idx = []
            subgraph_count = 0
            # ground_truth_feats = []

            for ii in range(len(adj_list)):
                # loop over different orderings
                adj_full = adj_list[ii]
                # adj_tril = np.tril(adj_full, k=-1)

                idx = -1
                for jj in range(0, num_nodes, S):
                    # loop over different subgraphs
                    # one subgraph for each node (each row in the adj) minus the stride

                    idx += 1

                    ### for each size-(jj+K) subgraph, we generate edges for the new block of K nodes
                    if jj + K > num_nodes:
                        break

                    if idx not in rand_idx:
                        continue

                    ### get graph for GNN propagation
                    ### Adj block contains the graph from start to the current row which is jj
                    adj_block = np.pad(
                        adj_full[:jj, :jj],
                        ((0, K), (0, K)),
                        "constant",
                        constant_values=1.0,
                    )  # assuming fully connected for the new block
                    adj_block = np.tril(adj_block, k=-1)
                    adj_block = adj_block + adj_block.transpose()
                    adj_block = torch.from_numpy(adj_block).to_sparse()
                    edges += [adj_block.coalesce().indices().long()]

                    ### get attention index
                    # exist node: 0
                    # newly added node: 1, ..., K
                    if jj == 0:
                        att_idx += [np.arange(1, K + 1).astype(np.uint8)]
                    else:
                        att_idx += [
                            np.concatenate(
                                [
                                    np.zeros(jj).astype(np.uint8),
                                    np.arange(1, K + 1).astype(np.uint8),
                                ]
                            )
                        ]

                    ### get node feature index for GNN input
                    # use inf to indicate the newly added nodes where input feature is zero
                    if jj == 0:
                        node_idx_feat += [np.ones(K) * np.inf]
                    else:
                        node_idx_feat += [
                            np.concatenate(
                                [np.arange(jj) + ii * N, np.ones(K) * np.inf]
                            )
                        ]

                    ### get node index for GNN output
                    idx_row_gnn, idx_col_gnn = np.meshgrid(
                        np.arange(jj, jj + K), np.arange(jj + K)
                    )
                    idx_row_gnn = idx_row_gnn.reshape(-1, 1)
                    idx_col_gnn = idx_col_gnn.reshape(-1, 1)
                    node_idx_gnn += [
                        np.concatenate([idx_row_gnn, idx_col_gnn], axis=1).astype(
                            np.int64
                        )
                    ]

                    ### get predict label
                    label += [
                        adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
                    ]

                    # ground_truth_feats.append(node_feats_list[ii][: jj + K, :])

                    subgraph_size += [jj + K]
                    subgraph_idx += [
                        np.ones_like(label[-1]).astype(np.int64) * subgraph_count
                    ]
                    subgraph_count += 1

            ### adjust index basis for the selected subgraphs
            cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
            for ii in range(len(edges)):
                edges[ii] = edges[ii] + cum_size[ii]
                node_idx_gnn[ii] = node_idx_gnn[ii] + cum_size[ii]

            ### pack tensors
            data = {}

            # print(f"node feat list: {node_feats_list}")

            if (self.node_feats is not None) and (self.has_nodes_feats is True):
                data["feats"] = np.stack(node_feats_list, axis=0)
                # data["feats"] = np.stack(ground_truth_feats, axis=0)

            data["adj"] = np.tril(np.stack(adj_list, axis=0), k=-1) # dense adjs (ground truth, num = number of canonical ordering)
            data["edges"] = torch.cat(edges, dim=1).t().long()
            data["node_idx_gnn"] = np.concatenate(node_idx_gnn)
            data["node_idx_feat"] = np.concatenate(node_idx_feat)
            data["label"] = np.concatenate(label)
            data["att_idx"] = np.concatenate(att_idx)
            data["subgraph_idx"] = np.concatenate(subgraph_idx)
            data["subgraph_count"] = subgraph_count
            data["num_nodes"] = num_nodes
            data["subgraph_size"] = subgraph_size
            data["num_count"] = sum(subgraph_size)
            data_batch += [data]

        end_time = time.time()

        return data_batch

    def __len__(self):
        return self.num_graphs

    def collate_fn(self, batch):
        assert isinstance(batch, list)
        start_time = time.time()
        batch_size = len(batch)
        N = self.max_num_nodes
        C = self.num_canonical_order
        batch_data = []

        for ff in range(self.num_fwd_pass): #ff=1
            data = {}
            batch_pass = []
            for bb in batch:
                batch_pass += [bb[ff]]



            pad_size = [self.max_num_nodes - bb["num_nodes"] for bb in batch_pass]
            subgraph_idx_base = np.array(
                [0] + [bb["subgraph_count"] for bb in batch_pass]
            )
            subgraph_idx_base = np.cumsum(subgraph_idx_base)

            data["subgraph_idx_base"] = torch.from_numpy(subgraph_idx_base)

            data["num_nodes_gt"] = (
                torch.from_numpy(np.array([bb["num_nodes"] for bb in batch_pass]))
                .long()
                .view(-1)
            )

            data["adj"] = torch.from_numpy(
                np.stack(
                    [
                        np.pad(
                            bb["adj"],
                            ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
                            "constant",
                            constant_values=0.0,
                        )
                        for ii, bb in enumerate(batch_pass)
                    ],
                    axis=0,
                )
            ).float()  # B X C X N X N



            if self.has_nodes_feats is True:
                data["feats"] = torch.from_numpy(
                    np.stack(
                        [
                            np.pad(
                                bb["feats"],
                                ((0, 0), (0, pad_size[ii]),(0, 0),(0,0)),
                                "constant",
                                constant_values=0.0,
                            )
                            for ii, bb in enumerate(batch_pass)
                        ],
                        axis=0,
                    )
                ).float()

                # temp=data["feats"]
                # print(f"collate_fn, data[feats].shape: {temp.shape}")

            idx_base = np.array([0] + [bb["num_count"] for bb in batch_pass])
            idx_base = np.cumsum(idx_base)

            data["edges"] = torch.cat(
                [bb["edges"] + idx_base[ii] for ii, bb in enumerate(batch_pass)], dim=0
            ).long()

            data["node_idx_gnn"] = torch.from_numpy(
                np.concatenate(
                    [
                        bb["node_idx_gnn"] + idx_base[ii]
                        for ii, bb in enumerate(batch_pass)
                    ],
                    axis=0,
                )
            ).long()

            data["att_idx"] = torch.from_numpy(
                np.concatenate([bb["att_idx"] for bb in batch_pass], axis=0)
            ).long()

            # shift one position for padding 0-th row feature in the model
            node_idx_feat = (
                np.concatenate(
                    [
                        bb["node_idx_feat"] + ii * C * N
                        for ii, bb in enumerate(batch_pass)
                    ],
                    axis=0,
                )
                + 1
            )
            node_idx_feat[np.isinf(node_idx_feat)] = 0
            node_idx_feat = node_idx_feat.astype(np.int64)
            data["node_idx_feat"] = torch.from_numpy(node_idx_feat).long()

            data["label"] = torch.from_numpy(
                np.concatenate([bb["label"] for bb in batch_pass])
            ).float()

            data["subgraph_idx"] = torch.from_numpy(
                np.concatenate(
                    [
                        bb["subgraph_idx"] + subgraph_idx_base[ii]
                        for ii, bb in enumerate(batch_pass)
                    ]
                )
            ).long()

            batch_data += [data]

        end_time = time.time()
        # print('collate time = {}'.format(end_time - start_time))

        return batch_data

