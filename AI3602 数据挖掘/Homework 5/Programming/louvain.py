# 100 - 120 lines of code in this file
import random
from collections import defaultdict, Counter

from graph import WeightedDiGraph
from community import Community

from typing import List, Dict, Tuple


class Louvain:
    def __init__(self, graph: WeightedDiGraph):
        # Number of nodes in the original graph
        # NOTE: These attributes should NOT be modified.
        self.original_graph = graph
        self.n_nodes = graph.N
        self.n_passes = 0

        # "G" is a graph of communities (we refer to it as "metagraph")
        # initially each node is its own community,
        # so the metagraph is (structurally) the same as the original graph
        self.G: WeightedDiGraph = graph.copy()

        # each node is initially a community of its own
        # we initialize n_nodes communities
        # the community is a map (community_id -> Community)
        self.communities: Dict[int, Community] = self._init_communities(
            self.G, self.n_nodes
        )

        # each node in the original graph belongs to a community,
        # this is tracked by node2commid (node_id -> community_id)
        # it maps a node in the original graph to its community id.
        self.node2commid: Dict[int, int] = {
            node_id: node_id for node_id in range(self.n_nodes)
        }

        # We use "metanode" to refer to a node in the metagraph
        # Each metanode represents a community of nodes in the original graph
        # In phase 1 of Louvain method (partitioning),
        # different metanodes might be merged into a single community
        # This is tracked by metanode2commid (metanode_id -> community_id)
        # it maps a metanode in the metagraph to its community id.
        # Initially, each metanode is its own community.
        self.metanode2commid: Dict[int, int] = {
            node_id: node_id for node_id in range(self.n_nodes)
        }

        # NOTE: In the code below we will use node and metanode interchangeably
        # unless otherwise stated, a "node" refers to a metanode in the metagraph

    def _init_communities(
        self, graph: WeightedDiGraph, n_communities: int
    ) -> Dict[int, Community]:
        """
        Initializes n_communities communities associated with the given graph.
        Each community is initialized with a single node.
        """
        return {
            node_id: Community(id=node_id, graph=graph, nodes={node_id})
            for node_id in range(n_communities)
        }

    def get_community_of_node(self, node: int) -> Community:
        """Returns the community to which the given node belongs."""
        return self.communities[self.metanode2commid[node]]

    def delta_modularity(self, node: int, community: Community) -> float:
        """Computes Delta_Q(i -> C),
        i.e., the change in modularity if we move node i to community C.
        """

        # TODO: (Task 3) Compute Delta_Q(i -> C), i.e.,
        # the change in modularity if we move node i to community C
        # <= 10 lines of code expected, depending on your coding style.
        #
        # Hints:
        # 1. The formula for Delta_Q(i -> C) is given in the handout.
        # 2. Make use of the functions you have implemented in the `Community` class.
        # 3. You can get the sum of edge weights of the metagraph by `self.G.M`.

        term_1 = community.node2comm_degree(node) / self.G.M
        term_2_a = self.G.get_in_degree(node) * community.get_out_degree()
        term_2_b = self.G.get_out_degree(node) * community.get_in_degree()
        term_2 = (term_2_a + term_2_b) / (self.G.M ** 2)
        delta_q = term_1 - term_2

        # End of TODO

        return delta_q

    def phase1(self):
        n_metanodes = self.G.N
        num_iter = 0
        modularity_gain = 0
        while True:
            num_iter += 1
            changed = False
            n_changed = 0

            # TODO (Optional): In practice, the order in which we visit
            # the nodes might affect the final result.
            # We visit the nodes in the order of their metanode ids,
            # you can also try different orders (e.g., randomly shuffling the nodes).
            node_iterator = list(range(n_metanodes))

            for metanode in node_iterator:
                # remove current node from its old community
                old_community = self.get_community_of_node(metanode)
                old_community.remove_node(metanode)

                # TODO (Task 4): Compute Delta_Q(C -> i) for old_community and metanode.
                # You should set the variable `delta_q_del` to the computed value.
                #
                # Hints:
                # 1. Delta_Q(C -> i) = -Delta_Q(i -> C)
                # 2. Only one line of code is required here.

                delta_q_del = -self.delta_modularity(metanode, old_community)

                # End of TODO

                best_modularity = 0
                best_community = old_community
                # Iterate over neighbors of the current node
                for nbr in sorted(self.G.get_neighbors(metanode)):
                    # get the community of the neighbor node
                    new_community = self.get_community_of_node(nbr)

                    # skip if the neighbor is in the same old community
                    if new_community.id == old_community.id:
                        continue

                    # TODO (Task 4):
                    # Compute Delta_Q(i -> C) for new_community and metanode.
                    # Compute Delta_Q = Delta_Q(C -> i) + Delta_Q(i -> C).
                    # Update best_modularity and best_community
                    # if the new community has a higher modularity gain.
                    # Around 5 lines of code expected.

                    delta_q_add = self.delta_modularity(metanode, new_community)
                    delta_q = delta_q_del + delta_q_add
                    if delta_q > best_modularity:
                        best_modularity = delta_q
                        best_community = new_community

                    # End of TODO

                # add current node to the best community
                self.metanode2commid[metanode] = best_community.id
                best_community.add_node(metanode)
                modularity_gain += best_modularity

                # Update the changed flag if the node has changed its community
                if best_community.id != old_community.id:
                    changed = True
                    n_changed += 1

            print(
                f"| Pass: {self.n_passes:3d} "
                f"| Phase 1 | Iter: {num_iter:3d} "
                f"| Nodes changed: {n_changed:5d} ({changed}) "
                f"| #Communities: {len(set(self.metanode2commid.values())):5d} "
                f"| Modularity gain: {modularity_gain:.4f} |"
            )

            if not changed:
                break

    def _update_node2commid(self):
        """Reassign nodes to their new communities after phase 1."""

        for node in range(self.n_nodes):
            # id of the old community of current node (in original graph)
            metanode_id = self.node2commid[node]
            # the metanode might be merged into other communities
            # so we need to find the new community id of the metanode
            community_id = self.metanode2commid[metanode_id]  # new community id
            # reassign the node to the new community
            self.node2commid[node] = community_id

    def _reindex_communities(self):
        """
        Reindex communities to make the community ids contiguous.
        Some communities might have been removed during phase 1,
        so we rearrange community ids so that they start from 0 and are contiguous

        E.g., if the communities are [0, 1, 3, 4, 5, 7, 8, 9],
        we reindex them to [0, 1, 2, 3, 4, 5, 6, 7]

        NOTE: `node2commid`, `metanode2commid` and `communities` will be updated.
        """
        remaining_communities = set(self.metanode2commid.values())
        reindex = {}
        for new_id, old_id in enumerate(remaining_communities):
            reindex[old_id] = new_id

        # update node2commid and metanode2commid to new community ids
        for node in range(self.n_nodes):
            self.node2commid[node] = reindex[self.node2commid[node]]
        for meta_node in range(self.G.N):
            self.metanode2commid[meta_node] = reindex[self.metanode2commid[meta_node]]

        # update community dict, drop removed communities
        self.communities = {
            reindex[old_id]: comm
            for old_id, comm in self.communities.items()
            if old_id in reindex
        }
        # update community id for consistency
        for new_id, comm in self.communities.items():
            comm.id = new_id

    def phase2(self):
        print(f"| Pass: {self.n_passes:3d} | Phase 2 | Building new graph. |")
        # update node to their new communities
        self._update_node2commid()
        # reindex communities to make the community ids continuous
        self._reindex_communities()

        new_edges: Dict[Tuple[int, int], int] = defaultdict(int)

        # TODO: (Task 4) Create a new metagraph of the updated communities
        # fill in `new_edges` with new edges between communities with updated weights
        # The format of `new_edges` is {(src, dst): weight}
        # 5-10 lines of code expected.
        #
        # Hints:
        # 1. The communities are now the new nodes in the new graph.
        # 2. The edges between the communities are the sum of the weights of the edges
        #    between the nodes in the original graph.

        for metanode in self.G.nodes:
            src = self.metanode2commid[metanode]
            for nbr in self.G.get_out_neighbors(metanode):
                dst = self.metanode2commid[nbr]
                weight = self.G.get_edge_weight(metanode, nbr)
                new_edges[(src, dst)] += weight

        # End of TODO

        edge_list = [(src, dst, weight) for (src, dst), weight in new_edges.items()]

        return WeightedDiGraph(edge_list)

    def louvain(self):
        random.seed(0)
        self.n_passes = 0
        while True:
            self.n_passes += 1

            self.phase1()
            g = self.phase2()

            # check if the new metagraph is the same as the old one
            if g.edges == self.G.edges:
                break

            # update the metagraph
            self.G = g
            self.communities = self._init_communities(self.G, self.G.N)
            self.metanode2commid = {node_id: node_id for node_id in range(self.G.N)}

        return self.node2commid

    def merge_communities(
        self,
        node2commid: Dict[int, int],
        n_expected_communities: int,
        gt_map: Dict[int, int],
    ):
        # TODO (Task 4): merge extra communities to reduce the number of communities.
        # Update the `node2commid` dictionary to merge the communities.
        #
        # Hints:
        # 1. You can use the provided labels for merging the communities,
        #    e.g., consider merging two communities if many nodes in the
        #    community share the same ground-truth label.
        # 2. Another strategy is to find and merge small communities
        #    according to the delta modularity before and after merging them.
        #
        # Other merging strategies are also acceptable. You are encouraged to
        # experiment with different strategies to improve the performance.
        # However, even if you do not merge any communities, you will still get
        # a base score as long as the core of your Louvain algo is correctly implemented.

        import numpy as np

        commid_list = sorted(set(node2commid.values()))
        label_list = sorted(set(gt_map.values()))
        cnt_matrix = np.zeros((len(commid_list), len(label_list)), dtype=int)
        for node, label in gt_map.items():
            commid = node2commid[node]
            cnt_matrix[commid, label] += 1

        reindex: Dict[int, int] = defaultdict(int)
        for label in label_list:
            commid = np.argmax(cnt_matrix[:, label])
            reindex[commid] = label
        for commid in commid_list:
            if commid not in reindex:
                reindex[commid] = np.argmax(cnt_matrix[commid, :])

        for node, commid in node2commid.items():
            node2commid[node] = reindex[commid]
        assert len(set(node2commid.values())) == n_expected_communities

        # End of TODO

        return node2commid
