import matplotlib.pyplot as plt
import networkx as nx
import pickle
import numpy as np

# from data_sort import get_whole_graph

# G_, A_, node_feats = get_whole_graph("Firenze", "DFS")
#
# # G가 없고 A만 있는 경우
# G_ = nx.from_numpy_array(A_)
#
# pos_dict = {}
#
# for i in range(len(node_feats[0])):
#     pos_dict[i] = (node_feats[0][i][0][0], node_feats[0][i][0][1])
#
# pos = pos_dict
# nx.draw_networkx_nodes(G_, pos, node_size=0.5, node_color='black')
# nx.draw_networkx_edges(G_, pos, alpha=0.5, width=1)
# plt.axis('off')
# plt.show()
# save_path = "./datasets_dev/cities/" + "Firenze" + ".png"
# plt.savefig(save_path)

with open ("output/3_base_recursive/out_graphs.pickle", "rb") as fr:
    data=pickle.load(fr)

t_A=data[0]
t_A=np.array(t_A[0])
print(f"t_A: {t_A}")
t_F=data[1]
t_F=np.array(t_F[0])
print(f"t_F: {t_F}")

G_1 = nx.from_numpy_array(t_A)
pos_dict = {}
print(len(t_F))

for i in range(len(t_F)):
    pos_dict[list(G_1.nodes)[i]] = (t_F[i][0], t_F[i][1])

pos = pos_dict
pos_2=nx.spring_layout(G_1)
print(f"pos : {pos}")
nx.draw_networkx_nodes(G_1, pos, node_size=0.5, node_color='black')
nx.draw_networkx_edges(G_1, pos, alpha=0.5, width=1)
# nx.draw_networkx_nodes(G_1, pos_2, node_size=0.5, node_color='black')
# nx.draw_networkx_edges(G_1, pos_2, alpha=0.5, width=1)
plt.axis('off')
plt.show()
save_path = "D:/LRG/datasets_dev/cities/norm/" + "Firenze_t1" + ".png"
plt.savefig(save_path)