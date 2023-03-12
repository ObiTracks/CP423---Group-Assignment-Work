import argparse
import numpy as np
import scipy.sparse as sp

# Define command line arguments
parser = argparse.ArgumentParser(description='PageRank algorithm for Stanford4 web graph')
parser.add_argument('--maxiteration', type=int, default=100, help='maximum number of iterations')
parser.add_argument('--lambda', type=float, default=0.15, dest='lmbda', help='λ parameter value')
parser.add_argument('--thr', type=float, default=1e-4, help='threshold value')
parser.add_argument('--nodes', type=int, nargs='+', default=None, help='NodeIDs to get PageRank values')

# Parse command line arguments
args = parser.parse_args()

# Load data from file
data = np.loadtxt('web-Stanford.txt', dtype=int)
num_nodes = max(max(data[:, 0]), max(data[:, 1])) + 1
indices = (data[:, 0], data[:, 1])
adjacency_matrix = sp.coo_matrix((np.ones_like(data[:, 0]), indices), shape=(num_nodes, num_nodes))
adjacency_matrix = adjacency_matrix.tocsr()

# Normalize adjacency matrix
out_degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
out_degrees[out_degrees == 0] = 1  # Avoid division by zero
transition_matrix = sp.diags(1 / out_degrees) @ adjacency_matrix

# Initialize PageRank vector with uniform distribution
page_rank = np.ones(num_nodes) / num_nodes

# PageRank algorithm
for i in range(args.maxiteration):
    previous_page_rank = page_rank.copy()
    page_rank = (1 - args.lmbda) * transition_matrix @ page_rank + args.lmbda / num_nodes
    difference = np.linalg.norm(page_rank - previous_page_rank, 1)
    if difference < args.thr:
        break

# Get PageRank values for specified nodes
if args.nodes:
    for node in args.nodes:
        print(f'Node {node}: {page_rank[node]:.6f}')
