import numpy as np
from pygsp import graphs

def make_simple_dataset(name, **kwargs):
    return make_cloud(name)


def make_cloud(name):
    if name.lower() == "grid2d":
        G = graphs.Grid2d(N1=8, N2=8)
    elif name.lower() == "ring":
        G = graphs.Ring(N=64)
    elif name.lower() == "bunny":
        G = graphs.Bunny()
    elif name.lower() == "airfoil":
        G = graphs.Airfoil()
    elif name.lower() == "minnesota":
        G = graphs.Minnesota()
    elif name.lower() == "sensor":
        G = graphs.Sensor(N=64)
    elif name.lower() == "community":
        G = graphs.Community(N=64)
    elif name.lower() == "barabasialbert":
        G = graphs.BarabasiAlbert(N=64)
    elif name.lower() == "davidsensornet":
        G = graphs.DavidSensorNet(N=64)
    elif name.lower() == "erdosrenyi":
        G = graphs.ErdosRenyi(N=64)
    else:
        raise ValueError("Unknown dataset: {}".format(name))

    if not hasattr(G, "coords"):
        G.set_coordinates(kind="spring")
    x = G.coords.astype(np.float32)
    y = np.zeros(x.shape[0])  # X[:,0] + X[:,1]
    A = G.W
    if A.dtype.kind == "b":
        A = A.astype("i")

    return A, x, G