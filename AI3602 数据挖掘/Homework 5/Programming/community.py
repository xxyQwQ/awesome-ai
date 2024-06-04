from graph import WeightedDiGraph
from typing import Set, Optional


class Community:
    def __init__(
        self,
        id: int,
        graph: WeightedDiGraph,
        nodes: Optional[Set[int]] = None,
    ):
        self.id = id
        self.graph = graph
        self.nodes = nodes if nodes is not None else set()
        self.in_degree = sum(self.graph.get_in_degree(node) for node in self.nodes)
        self.out_degree = sum(self.graph.get_out_degree(node) for node in self.nodes)

    def __repr__(self):
        return (
            f"Community(id={self.id}, nodes={self.nodes}, "
            f"in_degree={self.in_degree}, out_degree={self.out_degree})"
        )

    def add_node(self, node: int):
        """Adds a node to the community."""
        # We cache the in-degree and out-degree of the community
        # (\Sigma_tot^in, \Sigma_tot^out in the formula)
        # and update them whenever a node is added or removed.
        # This is because calculating the in-degree and out-degree of the community
        # by iterating over all nodes can be expensive when the community is large.

        # TODO: (Task 3) Add a node to the community.
        # You should update self.nodes, self.in_degree and self.out_degree accordingly.
        # Around 3 lines of code expected.
        #
        # Hint:
        # The `WeightedDiGraph` class provides utility functions
        # to get the in- and out-degrees of a node.

        self.nodes.add(node)
        self.in_degree += self.graph.get_in_degree(node)
        self.out_degree += self.graph.get_out_degree(node)

        # End of TODO

    def remove_node(self, node: int):
        """
        Removes a node from the community.
        Raises an error if the node is not in the community.
        """
        if node not in self.nodes:
            raise ValueError(f"Node {node} not in community {self.id}.")

        # TODO: (Task 3) Remove a node from the community.
        # You should update self.nodes, self.in_degree and self.out_degree accordingly.
        # Around 3 lines of code expected.
        #
        # Hint:
        # The `WeightedDiGraph` class provides utility functions
        # to get the in- and out-degrees of a node.

        self.nodes.remove(node)
        self.in_degree -= self.graph.get_in_degree(node)
        self.out_degree -= self.graph.get_out_degree(node)

        # End of TODO

    def get_in_degree(self) -> int:
        """Get the total in-degree of the community."""
        return self.in_degree

    def get_out_degree(self) -> int:
        """Get the total out-degree of the community."""
        return self.out_degree

    def node2comm_in_degree(self, node: int) -> int:
        """Returns the sum of weights of in-edges from the community to the node."""

        # TODO: (Task 3)
        # Compute and return the sum of in-edges weights from the community to the node.
        # The edges here are directed from the community to the node.
        # Around 5 lines of code expected.
        #
        # Hints:
        # 1. You can check the test cases in `test_modularity.py` for example outputs.
        # 2. The `WeightedDiGraph` class provides utility functions
        #    to get the in-/out- neighbors of a node and the weight of an edge.

        d_in = 0
        for nbr in self.graph.get_in_neighbors(node):
            if nbr in self.nodes:
                d_in += self.graph.get_edge_weight(nbr, node)

        # End of TODO

        return d_in

    def node2comm_out_degree(self, node: int) -> int:
        """Returns the sum of weights of out-edges from the node to the community."""

        # TODO: (Task 3)
        # Compute and return the sum of out-edge weights from the node to the community.
        # The edges here are directed from the node to the community.
        # Around 5 lines of code expected.
        #
        # Hints:
        # 1. You can check the test cases in `test_modularity.py` for example outputs.
        # 2. The `WeightedDiGraph` class provides utility functions
        #    to get the in-/out- neighbors of a node and the weight of an edge.

        ##################
        # Your Code Here #
        ##################

        d_out = 0
        for nbr in self.graph.get_out_neighbors(node):
            if nbr in self.nodes:
                d_out += self.graph.get_edge_weight(node, nbr)

        # End of TODO

        return d_out

    def node2comm_degree(self, node: int) -> int:
        """
        Returns the sum of weights of edges between the community and the node.
        I.e., returns the value of k_{i, in} in the formula of modularity gain.
        """
        return self.node2comm_in_degree(node) + self.node2comm_out_degree(node)
