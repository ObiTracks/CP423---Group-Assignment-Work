import argparse
import numpy as np
import scipy.sparse as sp


def load_data(file_name):
  data = np.loadtxt(file_name, dtype=int)
  num_nodes = max(max(data[:, 0]), max(data[:, 1])) + 1
  indices = (data[:, 0], data[:, 1])
  adjacency_matrix = sp.coo_matrix((np.ones_like(data[:, 0]), indices),
                                   shape=(num_nodes, num_nodes)).tocsr()
  return adjacency_matrix


def normalize_matrix(adjacency_matrix):
  out_degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
  out_degrees[out_degrees == 0] = 1  # Avoid division by zero
  transition_matrix = sp.diags(1 / out_degrees) @ adjacency_matrix
  return transition_matrix


def page_rank(transition_matrix,
              lmbda=0.15,
              max_iteration=100,
              threshold=1e-4):
  n = transition_matrix.shape[0]
  page_rank = np.ones(n) / n
  for i in range(max_iteration):
    previous_page_rank = page_rank.copy()
    page_rank = (1 - lmbda) * transition_matrix @ page_rank + lmbda / n
    difference = np.linalg.norm(page_rank - previous_page_rank, 1)
    if difference < threshold:
      break
  return page_rank




# Define command line arguments
parser = argparse.ArgumentParser(
  description='PageRank algorithm for Stanford4 web graph')
parser.add_argument('--maxiteration',
                    type=int,
                    default=100,
                    help='maximum number of iterations')
parser.add_argument('--lambda',
                    type=float,
                    default=0.15,
                    dest='lmbda',
                    help='λ parameter value')
parser.add_argument('--thr', type=float, default=1e-4, help='threshold value')
parser.add_argument('--nodes',
                    type=int,
                    nargs='+',
                    default=None,
                    help='NodeIDs to get PageRank values')


# Parse command line arguments
args = parser.parse_args()
# Load data from file
adjacency_matrix = load_data("web-Stanford.txt.gz")
print(adjacency_matrix[:10])
# Normalize adjacency matrix
transition_matrix = normalize_matrix(adjacency_matrix)
# Run PageRank algorithm
page_rank_values = page_rank(transition_matrix, args.lmbda, args.maxiteration,
                             args.thr)

# Get PageRank values for specified nodes
if args.nodes:
  for node in args.nodes:
    print(f'Node {node}: {page_rank_values[node]:.8f}')
