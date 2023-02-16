import networkx as nx

from simple_data import make_simple_dataset
import networkx as nx
from networkx import grid_2d_graph

G_1_A, G_1_X, G=make_simple_dataset("Grid2d")

G_2=grid_2d_graph(10,10)

print(type(G))
print(G)
print(type(G_2))
# print(G_2.number_of_nodes())
# print(nx.adjacency_matrix(G_2))
