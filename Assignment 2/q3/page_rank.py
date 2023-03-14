# Instructions to run
# python page_rank.py --maxiteration <int> -–lambda <float> –-thr <float> –-nodes <list of int>
# Default:
# python page_rank.py --maxiteration 20 -–lambda .25 –-thr .01 –-nodes [5,87524,632]

import argparse


def web_graph(node_count):
    web_graph = {i: [] for i in range(node_count)}
    return web_graph

def read_dataset():
    # Open the file for reading
    with open('web-Stanford.txt', 'r') as f:
        # Read the contents of the file line by line
        edge_count = 0

        node_count = None
        for line in f:
            # Process the line as needed
            line = line.strip()

            if line.startswith('# Nodes:'):
                node_count = int(line.split()[2])
                print("nodes:", node_count)
                wg = web_graph(node_count)

            else:
                # Split the line into two parts
                parts = line.split()
                if parts[0].isdigit():
                    # Extract the two node IDs as integers
                    node1 = int(parts[0])-1
                    node2 = int(parts[1])-1

                    wg[node1].append(node2)
                    edge_count += 1
    return wg, edge_count


def pagerank(wg, pr_lambda, maxiteration, thr, nodes):
    node_count = len(wg) - 1
    prev_iteration = [(1 / node_count) for i in range(node_count)]

    for iteration_count in range(maxiteration):
        new_iteration = []

        for node, neighbors in enumerate(wg):
            # Calculate the sum of all prev_iteration values for the neighbors
            sum_prev = 0
            for node2 in range(node_count):
                if node2 != node:
                    if len(wg[node2]) != 0:
                        sum_prev += (prev_iteration[node2] / len(wg[node2]))

            # Calculate the new_iteration value for this node
            new_pagerank = (pr_lambda / node_count) + (1 - pr_lambda) * sum_prev

            # Add the new_score to the new_iteration list
            new_iteration.append(new_pagerank)

        # Check for convergence
        converged = True
        for node in nodes:
            # Check if the PageRank score for this node has changed by less than thr
            if abs(prev_iteration[node] - new_iteration[node]) >= thr:
                converged = False
                break

        # If all nodes have converged, return the final PageRank scores
        if converged:
            return [new_iteration[node] for node in nodes]

        # Otherwise, update prev_iteration for the next iteration
        prev_iteration = new_iteration

    # Return the final PageRank scores after maxiteration iterations
    return [new_iteration[node] for node in nodes]


"""
if __name__ == '__main__':
    wg,ec = read_dataset()
    print("Web graph created")

    maxiteration = 20
    pr_lambda = .25
    thr = .01
    nodes = [5, 87524, 632]
    for node in nodes:
        node -= 1

    print(pagerank(wg, pr_lambda, maxiteration, thr, nodes))
    """

parser = argparse.ArgumentParser(
  description='PageRank algorithm for Stanford4 web graph')
parser.add_argument('--maxiteration',
                    type=int,
                    default=20,
                    help='maximum number of iterations')
parser.add_argument('--lambda',
                    type=float,
                    default=0.25,
                    dest='lmbda',
                    help='λ parameter value')
parser.add_argument('--thr', type=float, default=0.01, help='threshold value')
parser.add_argument('--nodes',
                    type=int,
                    nargs='+',
                    default=[5, 87524, 632],
                    help='NodeIDs to get PageRank values')


# Parse command line arguments
args = parser.parse_args()
# Load data from file
wg,ec = read_dataset()

# Run PageRank algorithm
page_rank_values = pagerank(wg, args.lmbda, args.maxiteration, args.thr, args.nodes)

for val in page_rank_values:
    print(f'Node {args.nodes[val]}: {page_rank_values[val]:.8f}')
