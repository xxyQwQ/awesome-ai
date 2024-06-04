# TuGraph Python APIs

> **[Official document](https://tugraph-db.readthedocs.io/en/latest/5.developer-manual/6.interface/3.procedure/4.Python-procedure.html)**.

This document lists some of TuGraph's Python APIs that you might need. For full document please refer to TuGraph's official document (linked above).

## `Galaxy`

> *A `Galaxy` is a TuGraph instance that holds multiple `GraphDB`s. A galaxy is stored in a directory and manages users and `GraphDB`s. Each `(user, GraphDB)` pair can have different access levels. You can use `db=Galaxy.OpenGraph(graph)` to open a graph.*

```py
galaxy = Galaxy(db_path)
galaxy.SetCurrentUser(username, password)

# open a graphdb
db = galaxy.OpenGraph(args.graph_name)

# ... do something with the db

# make sure to close the db and the galaxy
db.Close()
galaxy.Close()
```

## `GraphDB` and `Transaction`

> [Offical Doc (`GraphDB`)](https://tugraph-db.readthedocs.io/en/latest/5.developer-manual/6.interface/3.procedure/4.Python-procedure.html#liblgraph_python_api.GraphDB) | [Official Doc (`Transaction`)](https://tugraph-db.readthedocs.io/en/latest/5.developer-manual/6.interface/3.procedure/4.Python-procedure.html#liblgraph_python_api.Transaction)

> *A `GraphDB` stores the data about the graph, including labels, vertices, edges and indexes. Since Garbage Collection in Python is automatic, you need to close the DB with `GraphDB.Close()` at the end of its lifetime. Make sure you have either committed or aborted every transaction that is using the DB before you close the DB.*

In HW2 we only need to **read data from** the database. This can be done via a **(read) transaction**. Recall that (if you have enrolled in *AI3613 Database System Concepts*) a **transaction** is a collection of operations that performs a single logical function in a database.

All the operations for querying or updating the graph database are performed in transactions.

```py
# create a read transaction by
txn = db.CreateReadTxn()

# create a vertex iterator
vit = txn.GetVertexIterator()

# remember to commit the transaction after using it
txn.Commit()
```

#### `Transaction.GetVertexIterator()`

- `txn.GetVertexIterator()`. Returns a `VertexIterator` pointing to the first vertex in the graph.
- `txn.GetVertexIterator(node_id: int)`. Returns a `VertexIterator` pointing to the node `node_id`.
- `txn.GetVertexIterator(node_id: int, nearest: bool)`. Returns a `VertexIterator`. If `nearest == True`, then points to *the first vertex with `id >= node_id`*.

#### `Transaction.Commit()`

Commits the transaction, i.e., declare that you have done everything you need and close this transaction.

## `VertexIterator`

> [Official Doc (`VertexIterator`)](https://tugraph-db.readthedocs.io/en/latest/5.developer-manual/6.interface/3.procedure/4.Python-procedure.html#liblgraph_python_api.VertexIterator)

> *`VertexIterator` can be used to retrieve info of a vertex, or to scan through multiple vertices. Vertexes are sorted in ascending order of the their ids.*

Here we list some useful methods of the `VertexIterator` class. More can be found in the official document.

#### `VertexIterator.GetId() -> int`

Gets the integer id of the current vertex.

#### `VertexIterator.GetNumInEdges(n_limit: int) -> int`

Gets the number of in-coming edges of this vertex.

- `n_limit` (*Optional*) If provided, specifies the maximum number of edges to scan.

#### `VertexIterator.GetNumOutEdges(n_limit: int) -> int`

Gets the number of out-going edges of this vertex.

- `n_limit` (*Optional*) If provided, specifies the maximum number of edges to scan.

#### `VertexIterator.GetInEdgeIterator(eid: int) -> InEdgeIterator`

Gets an `InEdgeIterator` pointing to the in-edges of this vertex.

- `eid` (*Optional*) If provided, returns an iterator that points to the edge with `edge_id == eid`. Otherwise returns an iterator that points to the first in-edge.

#### `VertexIterator.GetOutEdgeIterator(eid: int) -> OutEdgeIterator`

Gets an `OutEdgeIterator` pointing to the out-edges of this vertex.

- `eid` (*Optional*) If provided, returns an iterator that points to the edge with `edge_id == eid`. Otherwise returns an iterator that points to the first out-edge.

#### `VertexIterator.Goto(vid: int, nearest: bool) -> bool`

Moves the iterator to the vertex specified by `vid`.

- `vid` Id of the target node.
- `nearest` (*Optional*) If `nearest == True`, then goes to the nearest vertex with `vertex_id >= vid`.

#### `VertexIterator.Next() -> bool`

Goes to the next vertex with `vertex_id > current_vid`.

#### `VertexIterator.ListDstVids(n_limit: int) -> Tuple[List[int], bool]`

Lists all destination vertex ids of the out-edges. The first element in the returned tuple is a list of destination vertex ids. The second element is a boolean indicating whether `n_limit` is exceeded.

- `n_limit` (*Optional*) Specifies the maximum number of destination vids to return.

#### `VertexIterator.ListSrcVids(n_limit: int) -> Tuple[List[int], bool]`

Lists all source vertex ids of the in-edges. The first element in the returned tuple is a list of source vertex ids. The second element is a boolean indicating whether `n_limit` is exceeded.

- `n_limit` (*Optional*) Specifies the maximum number of source vids to return.

#### `VertexIterator.IsValid() -> bool`

Tells whether the current iterator is still valid.

## `InEdgeIterator` and `OutEdgeIterator`

> [Official Doc (`InEdgeIterator`)](https://tugraph-db.readthedocs.io/en/latest/5.developer-manual/6.interface/3.procedure/4.Python-procedure.html#liblgraph_python_api.InEdgeIterator) | [Official Doc (`OutEdgeIterator`)](https://tugraph-db.readthedocs.io/en/latest/5.developer-manual/6.interface/3.procedure/4.Python-procedure.html#liblgraph_python_api.OutEdgeIterator)

These two iterators are similar to the vertex iterator, but they iterate over edges.

## Example

Here we provide a simple example that counts the number of vertices and edges of a graph

```py
txn = db.CreateReadTxn()  # creates read transaction
n_vertices = 0
n_edges = 0

vit = txn.GetVertexIterator()  # creates a vertex iterator

# iterate over all vertices
while vit.IsValid():
    n_vertices += 1

    # create an out-edge iterator
    eit = vit.GetOutEdgeIterator()

    # iterate over all out-edges of current vertex
    while eit.IsValid():
        n_edges += 1
        # move to next edge
        eit.Next()
    
    # move to next vertex
    vit.Next()
```
