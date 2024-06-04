from collections import defaultdict
from typing import List, Dict, Set, Tuple, Iterable


class WeightedDiGraph:
    """A weighted directed graph, used for Louvain community detection."""

    def __init__(self, edge_list: List[Tuple[int, int, int]]):
        """Creates a weighted directed graph instance from a list of edges.

        Args:
            edge_list (List[Tuple[int, int, int]]): A list of edges, where each edge
            is represented by a tuple (src, dst, weight).
        """

        # node_id -> neighbors going into the node
        self.in_neighbors: Dict[int, Set[int]] = defaultdict(set)
        # node_id -> neighbors going out of the node
        self.out_neighbors: Dict[int, Set[int]] = defaultdict(set)
        # (src, dst) -> edge weight
        self.edges: Dict[Tuple[int, int], float] = defaultdict(float)

        # node_id -> in-degree of the node
        self.in_degree: Dict[int, int] = defaultdict(int)
        # node_id -> out-degree of the node
        self.out_degree: Dict[int, int] = defaultdict(int)

        self.nodes = set()

        for src, dst, weight in edge_list:
            if (src, dst) in self.edges:
                print(
                    f"[!] Duplicate edge: ({src}, {dst}). "
                    "Will add new edge weights to existing edges."
                )
            self.add_edge(src, dst, weight)

        self.M = sum(self.edges.values())  # total edge weight
        self.N = len(self.nodes)  # total number of nodes

    def add_node(self, node: int):
        """Adds a node to the graph."""
        self.nodes.add(node)

    def add_edge(self, src: int, dst: int, weight: int = 1):
        """Adds an edge (src, dst) with the given weight to the graph.
        If the edge already exists, the weight is updated by adding the new weight.

        This function also updates the in- and out-degree of the nodes.

        Args:
            src (int): Source node.
            dst (int): Destination node.
            weight (int, optional): Weight of the edge. Defaults to 1.
        """
        self.add_node(src)
        self.add_node(dst)

        self.in_neighbors[dst].add(src)
        self.out_neighbors[src].add(dst)
        self.edges[(src, dst)] += weight

        self.in_degree[dst] += weight
        self.out_degree[src] += weight

    def get_neighbors(self, node: int, remove_duplicates: bool = True) -> Iterable[int]:
        """Returns the neighbors (both in- and out-neighbors) of a node.

        Args:
            node (int): Node ID.
            remove_duplicates (bool, optional): Removes duplicated neighbors if is True.
                If the neighbor is both an in- and out-neighbor, it will count only once.
                Defaults to True.

        Returns:
            Iterable[int]: A set (or list) of neighbors, depending on remove_duplicates.
        """
        if remove_duplicates:
            return set.union(self.get_in_neighbors(node), self.get_out_neighbors(node))

        return list(self.get_in_neighbors(node)) + list(self.get_out_neighbors(node))

    def get_in_neighbors(self, node: int) -> Set[int]:
        """Returns the in-neighbors of a node.
        I.e., the nodes that have an edge pointing to the node.

        Args:
            node (int): Node ID.

        Returns:
            Set[int]: A set of in-neighbors.
        """
        if node not in self.in_neighbors:
            return set()
        return self.in_neighbors[node]

    def get_out_neighbors(self, node: int) -> Set[int]:
        """Returns the out-neighbors of a node.
        I.e., the nodes for which the node has an edge pointing to them.

        Args:
            node (int): Node ID.

        Returns:
            Set[int]: A set of out-neighbors.
        """
        if node not in self.out_neighbors:
            return set()
        return self.out_neighbors[node]

    def has_edge(self, src: int, dst: int) -> bool:
        """Returns True if the edge (src, dst) exists in the graph."""
        return (src, dst) in self.edges

    def get_edge_weight(self, src: int, dst: int) -> float:
        # check if the edge exists
        if not self.has_edge(src, dst):
            if self.has_edge(dst, src):
                raise KeyError(
                    f"Edge {src} -> {dst} not found, did you mean {dst} -> {src}?"
                )
            else:
                raise KeyError(f"Edge {src} -> {dst} not found.")
        return self.edges[(src, dst)]

    def get_in_degree(self, node: int) -> int:
        """Returns the in-degree of a node."""
        if node not in self.in_degree:
            return 0
        return self.in_degree[node]

    def get_out_degree(self, node: int) -> int:
        """Returns the out-degree of a node."""
        if node not in self.out_degree:
            return 0
        return self.out_degree[node]

    def get_degree(self, node: int) -> int:
        """Return the degree of a node (in + out)."""
        return self.get_in_degree(node) + self.get_out_degree(node)

    def copy(self) -> "WeightedDiGraph":
        """Returns a deep copy of the graph."""
        return WeightedDiGraph(
            [(src, dst, weight) for (src, dst), weight in self.edges.items()]
        )

    @classmethod
    def from_csv_edges(
        cls, csv_path: str, has_header: bool = True
    ) -> "WeightedDiGraph":
        """
        Builds a weighted directed graph from a CSV file containing edges.
        The CSV file should contain two or three columns: src, dst, and (optional) weight.
        """
        edge_list = []
        with open(csv_path, "r", encoding="utf-8") as fi:
            if has_header:
                _ = fi.readline()  # skip csv header
            for line in fi.readlines():
                entries = line.strip().split(",")
                if len(entries) == 2:
                    src, dst = map(int, entries)
                    weight = 1
                elif len(entries) == 3:
                    src, dst, weight = map(int, entries)
                else:
                    raise ValueError(f"Invalid edge entry: {entries}")
                edge_list.append((src, dst, weight))

        return cls(edge_list)


if __name__ == "__main__":
    g = WeightedDiGraph.from_csv_edges("./p2_data/test_graph.csv")
    vertices = list(range(10))
    # (in, out)
    gt_degs = [
        (10, 20),
        (5, 70),
        (65, 15),
        (30, 30),
        (35, 10),
        (30, 5),
        (20, 0),
        (20, 10),
        (20, 30),
        (0, 45),
    ]
    for v, (in_deg, out_deg) in zip(vertices, gt_degs):
        assert g.get_in_degree(v) == in_deg
        assert g.get_out_degree(v) == out_deg

    assert g.get_in_neighbors(1) == set([8])
    assert g.get_in_neighbors(2) == set([0, 1, 3, 8, 9])
    assert g.get_out_neighbors(1) == set([2, 4, 5, 6, 7])
    assert g.get_out_neighbors(2) == set([3, 4])
    assert g.get_in_neighbors(9) == set()

    assert g.get_edge_weight(1, 2) == 10
    assert g.M == 235  # total edge weight
    assert g.N == 10  # total number of nodes

    g2 = g.copy()
    assert g2.M == g.M
    assert g2.N == g.N
    assert g2.edges == g.edges
    assert g2.in_neighbors == g.in_neighbors
    assert g2.out_neighbors == g.out_neighbors
    assert g2.in_degree == g.in_degree
    assert g2.out_degree == g.out_degree

    g2.add_edge(0, 1, 1)
    assert g2.has_edge(0, 1)
    assert not g.has_edge(0, 1)
    assert g2.get_edge_weight(0, 1) == 1

    print("Graph tests passed!")
