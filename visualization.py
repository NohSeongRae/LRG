import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

from dataloader_ import get_whole_graph

G_, A_, node_feats = get_whole_graph("Firenze", "BFS")

pos_dict = {}

for i in range(len(node_feats[0])):
    pos_dict[i] = (node_feats[0][i][0][0], node_feats[0][i][0][1])

pos = pos_dict
nx.draw_networkx_nodes(G_, pos, node_size=0.5, node_color='black')
nx.draw_networkx_edges(G_, pos, alpha=0.5, width=1)
plt.axis('off')
plt.show()
save_path = "./datasets/cities/" + "Firenze" + ".png"
plt.savefig(save_path)

