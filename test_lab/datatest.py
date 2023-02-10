import os
import networkx as nx
import numpy as np
from pygsp import graphs

G=graphs.Grid2d(N1=8, N2=8)
print(f'G: {G}')
if not hasattr(G, "coords"):
    G.set_coordinates(kind="spring")

x=G.coords.astype(np.float32)
# print(f"x: {x}")
# print(f"x.shape[0]: {x.shape[0]}")
# y=np.zeros(x.shape[0])
# print(f"y: {y}")
A=G.W
# print(f"A: {A}")

L=[A]
# print(f"L: {L}")
A_out=[]
S_out=[]

X=np.array(x)
print(f"X : {X}")
#


