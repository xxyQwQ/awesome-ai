import random
import torch
import torch.sparse
import numpy as np
import networkx as nx


class BiasedRandomWalker:
    """
    A biased random walker for generating random walks on a graph.
    """

    def __init__(self, data, p: float = 3.0, q: float = 2.0, p_inv: float = 1.0):
        self.data = data
        self.ret_p = p
        self.io_q = q
        self.inv_prob = p_inv

        self.edge_index = self.data.edge_index
        self.num_nodes = self.data.num_nodes
        
        self.adj_list, self.adj_list_inv = self._convert_to_adj_list()
        self.connected_nodes = list(range(self.num_nodes)) # count only nodes with out edges to walk from
        print(f"Number of connected nodes: {len(self.connected_nodes)}")
    
    
    def _convert_to_sparse_matrix(self):
        """Convert edge_index to a sparse adjacency matrix."""
        row, col = self.edge_index
        adj = torch.sparse_coo_tensor(torch.stack([row, col]), torch.ones_like(row), (self.num_nodes, self.num_nodes))
        return adj
    
    
    def _convert_to_adj_list(self):
        """Convert edge_index to an adjacency list."""
        adj_list = {}
        adj_list_inv = {}
        row, col = np.array(self.edge_index)

        for src, dst in zip(row, col):
            if src not in adj_list:
                adj_list[src] = []
            adj_list[src].append(dst)
            if dst not in adj_list_inv:
                adj_list_inv[dst] = []
            adj_list_inv[dst].append(src)
            
        return adj_list, adj_list_inv


    def _convert_to_networkx(self):
        """Convert edge_index to a networkx graph."""
        G = nx.DiGraph()
        row, col = self.edge_index
        edges = zip(row.tolist(), col.tolist())
        G.add_edges_from(edges)
        return G
    

    def _normalize(self, weights):
        """Normalizes the weights to make them sum to 1."""
        weights = np.array(weights)
        tot = weights.sum()
        return weights / tot


    def get_probs_uniform(self, curr_node: int):
        """Returns a normalized uniform probability distribution
        over the neighbors of the current node.
        """
        nexts = self.adj_list.get(curr_node, []) + self.adj_list_inv.get(curr_node, [])
        if not nexts:
            return [], []

        probs = [1 / len(nexts)] * len(nexts)
        return nexts, probs
    
    
    def get_probs_biased(self, curr_node: int, prev_node: int):
        """Returns a normalized biased probability distribution
        over the neighbors of the current node.
        """
        curr_nbrs = self.adj_list.get(curr_node, [])
        curr_nbrs_inv = self.adj_list_inv.get(curr_node, [])
        assert len(curr_nbrs) + len(curr_nbrs_inv) > 0, f"Node {curr_node} has no neighbors"

        prev_nbrs = self.adj_list.get(prev_node, []) + self.adj_list_inv.get(prev_node, [])
        # this is becasue curr node can walk along or inverse from prev node
        nexts = []
        unnormalized_probs = []

        for next_node in curr_nbrs:
            nexts.append(next_node)
            if next_node == prev_node: # return
                unnormalized_probs.append(1 / self.ret_p)
            elif next_node in prev_nbrs:
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(1 / self.io_q) # explore

        for next_node in curr_nbrs_inv:
            nexts.append(next_node)
            if next_node == prev_node:
                unnormalized_probs.append(1 / self.ret_p * 1 / self.inv_prob)
            elif next_node in prev_nbrs:
                unnormalized_probs.append(1 * 1 / self.inv_prob)
            else:
                unnormalized_probs.append(1 / self.io_q * 1 / self.inv_prob)

        # Normalize the probabilities
        probs = self._normalize(unnormalized_probs)
        return nexts, probs
    
    
    def walk(self, start: int, length: int):
        """Perform a random walk of length `length`, starting from node `start`.

        Args:
            start (int): The node id to start the random walk.
            length (int): The length of the random walk.

        Returns:
            List[int]: A list of node ids representing the random walk trajectory.
        """

        trace = [start]
        current_len = 1
        prev = None

        while current_len < length:
            curr_node = trace[-1]

            if prev is None:
                # For the first node, sample uniformly at random
                nexts, probs = self.get_probs_uniform(curr_node)
            else:
                # For the subsequent nodes, sample based on the biased probabilities
                nexts, probs = self.get_probs_biased(curr_node, prev)

            target = random.choices(nexts, probs)[0]
            trace.append(target)

            prev = curr_node
            current_len += 1

        return trace



if __name__ == "__main__":
    from dataset.dataloader import arxiv_dataset
    
    data = arxiv_dataset()
    walker = BiasedRandomWalker(data['graph'])

    start_node = data['train_idx'][222].item()
    walk_length = 10
    random_walk = walker.walk(start_node, walk_length)
    print("Random walk:", random_walk)
