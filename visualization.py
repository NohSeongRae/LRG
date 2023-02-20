import matplotlib.pyplot as plt
import networkx as nx

from data_sort import get_whole_graph

G_, adj_, feature_list = get_whole_graph("Firenze")

G_ = nx.from_numpy_array(adj_)

pos_dict = {}

for i in range(len(feature_list)):
    pos_dict[list(G_.nodes)[i]] = (feature_list[i][0], feature_list[i][1])



pos = pos_dict
nx.draw_networkx_nodes(G_, pos, node_size=0.5, node_color='black')
nx.draw_networkx_edges(G_, pos, alpha=0.5, width=1)
plt.axis('off')
plt.show()
save_path = "./datasets/cities/" + "Firenze" + ".png"
plt.savefig(save_path)
