import random
from typing import List, Tuple


class BiasedRandomWalker:
    """
    A biased random walker for generating random walks on a graph.

    Args:
    - db: A TuGraph database instance.
    - p (float, optional): The return parameter. Defaults to 1.2.
    - q (float, optional): The in-out parameter. Defaults to 2.0.
    """

    def __init__(self, db, p: float = 1.2, q: float = 2.0):
        self.db = db
        self.ret_p = p
        self.io_q = q

        self.connected_nodes = self._get_connected_nodes()

    def _get_connected_nodes(self):
        """
        Returns a list of nodes that have at least one edge connected to them.

        The dataset contains some isolated nodes,
        i.e., nodes with no edges connected to them.

        Since we cannot perform random walk on these isolated nodes,
        we manually exclude them from the list of connected nodes.

        XXX: Do NOT change this function.
        """
        txn = self.db.CreateReadTxn()
        vit = txn.GetVertexIterator()

        connected_nodes = []
        while vit.IsValid():
            if vit.GetNumOutEdges()[0] > 0:
                connected_nodes.append(vit.GetId())
            vit.Next()

        txn.Commit()
        return connected_nodes

    def _normalize(self, weights):
        """Normalizes the weights to make them sum to 1."""
        tot = sum(weights)
        return [p / tot for p in weights]

    def get_probs_uniform(self, txn, vit) -> Tuple[List[int], List[float]]:
        """Returns a normalized uniform probability distribution
        over the neighbors of the current node (i.e., the node pointed by `vit`)

        NOTE: This function returns a tuple of two lists:

        - List of neighbor node IDs
        - List of probabilities corresponding to the neighbor nodes

        XXX: Do NOT change the signature and return format of this function.
             This function will be used for automated grading.
        """
        nexts = vit.ListDstVids()[0]
        probs = [1 / len(nexts)] * len(nexts)
        return nexts, probs

    def get_probs_biased(self, txn, vit, prev: int) -> Tuple[List[int], List[float]]:
        """Returns a normalized biased probability distribution
        over the neighbors of the current node (i.e., the node pointed by `vit`)

        NOTE: This function returns a tuple of two lists:

        - List of neighbor node IDs
        - List of probabilities corresponding to the neighbor nodes

        XXX: Do NOT change the signature and return format of this function.
             This function will be used for automated grading.
        """

        # TODO: (Task 2)
        # Get the current neighbors of the node pointed by `vit`.
        # 1 line of code expected.
        curr_nbrs = vit.ListDstVids()[0]
        prev_nbrs = txn.GetVertexIterator(prev).ListDstVids()[0]

        nexts = []
        unnormalized_probs = []
        for next in curr_nbrs:
            nexts.append(next)

            # TODO: (Task 2)
            # Compute the unnormalized transition probs for the biased random walk.
            # For each neighbor node `next` of `curr`, compute the unnormalized probablity
            # of moving to `next` from `curr`, given we came from a previous node `prev`.
            # Append the unnormalized probability to the list `unnormalized_probs`.
            # Around 10 lines of code expected.
            #
            # Hints:
            # 1. The unnormalized probability should take values from [1, 1/p, 1/q],
            #    depending on the relationship between `prev`, `curr` and `next`.
            # 2. Use TuGraph's APIs to get the neighboring nodes.

            if next == prev:
                prob = 1.0 / self.ret_p
            elif next in prev_nbrs:
                prob = 1.0
            else:
                prob = 1.0 / self.io_q
            unnormalized_probs.append(prob)

            # End of TODO

        # normalize the probabilities
        probs = self._normalize(unnormalized_probs)
        return nexts, probs

    def walk(self, start: int, length: int) -> List[int]:
        """Perform a random walk of length `length`, starting from node `start`.

        Args:
            start (int): The node id to start the random walk.
            length (int): The length of the random walk.

        Returns:
            List[int]: A list of node ids representing the random walk trajectory.
        """

        # initiate a transaction
        txn = self.db.CreateReadTxn()
        # get a vertex iterator pointing to the start node
        vit = txn.GetVertexIterator(start)

        trace = [vit.GetId()]
        current_len = 1
        while current_len < length:

            # TODO (Task 2)
            # Implement the biased random walk.
            # Fill the list `trace` with the node ids of the random walk trajectory.
            # Around 10 lines of code expected.
            #
            # Hints:
            # 1. For the first node, there is no previous node, sample uniform at random.
            #    Use `get_probs_uniform`, which we have provided for you.
            # 2. For the subsequent nodes, sample based on the biased probabilities.
            #    Use `get_probs_biased`, which you have just implemented.

            if current_len == 1:
                nexts, probs = self.get_probs_uniform(txn, vit)
            else:
                prev = trace[-2]
                nexts, probs = self.get_probs_biased(txn, vit, prev)

            # `target` is to be sampled from neighboring nodes based on probabilities
            target = random.choices(nexts, probs)[0]
            trace.append(target)

            # End of TODO

            # move the iterator to the next node for a new iteration
            vit.Goto(vid=target, nearest=False)
            current_len += 1

        txn.Commit()
        return trace
