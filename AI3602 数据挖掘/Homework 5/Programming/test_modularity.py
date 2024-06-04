import unittest
from community import Community
from graph import WeightedDiGraph
from louvain import Louvain


def _get_graph():
    TEST_GRAPH_FILE = "./p2_data/test_graph.csv"
    return WeightedDiGraph.from_csv_edges(TEST_GRAPH_FILE)


class TestCommunity(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.graph = _get_graph()

    def test_add_node(self):
        # This function tests Community.add_node
        # TODO (optional): Add more test cases to cover your implementation
        # E.g., test adding more nodes to the community
        #       or test adding nodes to an initially empty community
        c = Community(id=0, graph=self.graph, nodes={0})
        c.add_node(9)
        self.assertEqual(c.get_in_degree(), 10 + 0)
        self.assertEqual(c.get_out_degree(), 20 + 45)

    def test_remove_node(self):
        # This function tests Community.remove_node
        # TODO (optional): Add more test cases to cover your implementation
        # E.g., test removing more nodes from the community until it's empty
        c = Community(id=0, graph=self.graph, nodes={1, 2, 9})
        c.remove_node(9)
        self.assertEqual(c.get_in_degree(), 5 + 65)
        self.assertEqual(c.get_out_degree(), 70 + 15)


class TestNode2CommDegree(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.graph = _get_graph()

    # This function tests Community.node2comm_in_degree
    # when the community is empty
    def test_node2comm_in_empty(self):
        # empty community
        c = Community(id=0, graph=self.graph, nodes=set())
        self.assertEqual(c.node2comm_in_degree(0), 0)
        self.assertEqual(c.node2comm_in_degree(1), 0)

    # This function tests Community.node2comm_in_degree
    # when the community contains one ore more nodes
    def test_node2comm_in(self):
        # single node
        # TODO (optional): Add more test cases to cover your implementation
        # E.g., test more neighboring nodes of this community
        c = Community(id=0, graph=self.graph, nodes={2})
        self.assertEqual(c.node2comm_in_degree(0), 0)
        self.assertEqual(c.node2comm_in_degree(3), 5)

        # multiple nodes
        # TODO (optional): Add more test cases to cover your implementation
        # E.g., test more neighboring nodes of this community
        c = Community(id=0, graph=self.graph, nodes={0, 2, 3})
        self.assertEqual(c.node2comm_in_degree(1), 0)
        self.assertEqual(c.node2comm_in_degree(4), 15)

    # This function tests Community.node2comm_out_degree
    # when the community is empty
    def test_node2comm_out_empty(self):
        # empty community
        c = Community(id=0, graph=self.graph, nodes=set())
        self.assertEqual(c.node2comm_out_degree(0), 0)
        self.assertEqual(c.node2comm_out_degree(1), 0)

    # This function tests Community.node2comm_out_degree
    # # when the community contains one ore more nodes
    def test_node2comm_out(self):
        # single node
        # TODO (optional): Add more test cases to cover your implementation
        # E.g., test more neighboring nodes of this community
        c = Community(id=0, graph=self.graph, nodes={2})
        self.assertEqual(c.node2comm_out_degree(4), 0)
        self.assertEqual(c.node2comm_out_degree(0), 5)

        # multiple nodes
        # TODO (optional): Add more test cases to cover your implementation
        # E.g., test more neighboring nodes of this community
        c = Community(id=0, graph=self.graph, nodes={0, 2, 3})
        self.assertEqual(c.node2comm_out_degree(4), 0)
        self.assertEqual(c.node2comm_out_degree(1), 10)

    def test_node2comm_empty(self):
        # empty community
        c = Community(id=0, graph=self.graph, nodes=set())
        self.assertEqual(c.node2comm_degree(0), 0)
        self.assertEqual(c.node2comm_degree(1), 0)

    def test_node2comm(self):
        # single node
        c = Community(id=0, graph=self.graph, nodes={2})
        self.assertEqual(c.node2comm_degree(0), 5)
        self.assertEqual(c.node2comm_degree(3), 20)
        self.assertEqual(c.node2comm_degree(4), 10)

        # multiple nodes
        c = Community(id=0, graph=self.graph, nodes={0, 2, 3})
        self.assertEqual(c.node2comm_degree(1), 10)
        self.assertEqual(c.node2comm_degree(4), 15)
        self.assertEqual(c.node2comm_degree(2), 25)
        self.assertEqual(c.node2comm_degree(9), 25)

    def test_self_loop(self):
        c = Community(id=0, graph=self.graph, nodes={3})
        self.assertEqual(c.node2comm_in_degree(3), 10)
        self.assertEqual(c.node2comm_out_degree(3), 10)


class TestDeltaModularity(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.graph = _get_graph()
        self.louvain = Louvain(self.graph)

    # This function tests Louvain.delta_modularity
    def test_delta_modularity(self):
        comm = self.louvain.communities[1]  # Community({1})
        comm.remove_node(1)  # make it empty

        # D(1 -> {})
        dq = self.louvain.delta_modularity(1, comm)
        self.assertAlmostEqual(dq, 0.0)

        # TODO (optional): Add more test cases to cover your implementation
        # E.g., add some node x into the community and calculate D(x -> comm)
        #       use assertAlmostEqual to test against the expected value
        #       you can derive the expected value manually or use `networkx`
        # NOTE: `networkx` APIs are allowed ONLY for writing test cases
        #       they should NOT be used in your main implementation

        # Here is an example of checking D(2 -> {1})
        # NOTE: The GT here is computed manually,
        #       you could also consider using `networkx.communities.modularity()`
        #       but we reiterate that `networkx` is only allowed for writing test cases
        comm.add_node(1)  # Add node 1 so that comm = {1}
        dq = self.louvain.delta_modularity(2, comm)
        self.assertAlmostEqual(dq, -0.04119511090991399)


if __name__ == "__main__":
    unittest.main()
