from dataloader import get_graph

def create_graphs(city_name, data_dir="data"):
    ## load datasets
    graphs = []
    feats = []

    G, X = get_graph(city_name)

    pos_dict = {}

    for i in range(len(X[0])):
        pos_dict[i] = (X[0][i][0][0], X[0][i][0][1])

    graphs.append(G)

    return graphs, feats