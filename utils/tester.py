import matplotlib.pyplot as plt
from level4.data_sort import *

graphs, adj_, feats =get_whole_graph_4("Firenze")

print(f"adj_: {adj_}\n")
print(f"feats: {feats}\n")

pos_dict = {}

for i in range(len(feats)):
    pos_dict[list(graphs.nodes)[i]] = (feats[i][0], feats[i][1])

pos = pos_dict
pos_2=nx.spring_layout(graphs)
print(f"pos : {pos}")
nx.draw_networkx_nodes(graphs, pos, node_size=0.5, node_color='black')
nx.draw_networkx_edges(graphs, pos, alpha=0.5, width=1)
# nx.draw_networkx_nodes(G_1, pos_2, node_size=0.5, node_color='black')
# nx.draw_networkx_edges(G_1, pos_2, alpha=0.5, width=1)
plt.axis('off')
plt.show()
# save_path = "D:/LRG/datasets_dev/cities/norm/" + "Firenze_t1" + ".png"
# plt.savefig(save_path)