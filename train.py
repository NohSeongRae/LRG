import argparse
import tensorflow as tf
from trainer import results_to_file, run_experiment
from blocks import NDP
from plot import plot_result

# initialize args
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="Grid2d")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--patience", type=int, default=1000)
parser.add_argument("--tol", type=float, default=1e-6)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()


# preprocess for NDP
def preprocess(X, A):
    return X, A


# pooling sequence - this implementation allows eigsh during pre-processing
def pooling(X, A):
    """

    :param X: node feature matrix
    :param A: adjacency matrix
    :return: A: initial adj matrix, X: node feature matrix, A_out[0]: pooled adj matrix, S_out[0]: Selection matrix
    """
    _, L = preprocess(X, A)
    A_out, S_out = NDP([L], 1)
    return A, X, A_out[0], S_out[0]


results = run_experiment(
    name=args.name,
    method="NDP",
    pooling=pooling,
    learning_rate=args.lr,
    es_patience=args.patience,
    es_tol=args.tol,
    runs=args.runs,
)

results_to_file(args.name, "NDP", *results)

plot_result()