def bfs_path_search(db, src: int, dst: int):
    """
    Finds the shortest path on a non-weighted graph
    from src to dst using breadth-first search.
    """
    # Create a read transaction to read the graph
    txn = db.CreateReadTxn()

    # store results
    prev = {}

    # bfs init
    q = []
    q.append(src)
    visited = set([src])

    prev[src] = -1

    # bfs
    while len(q):
        node = q.pop(0)
        visited.add(node)

        # we need a vertex iterator to access nodes
        vit = txn.GetVertexIterator(node)

        # use ListDstVids() to get the neighbors of the node
        nbr_list = vit.ListDstVids()[0]

        for nbr in nbr_list:
            if nbr not in visited:
                visited.add(nbr)
                prev[nbr] = node
                q.append(nbr)

    # print the path
    path = []
    while True:
        path.append(dst)
        dst = prev[dst]
        if dst == -1:
            break

    # close the transaction
    txn.Commit()

    return path[::-1]
