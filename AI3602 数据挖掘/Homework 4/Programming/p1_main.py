from argparse import ArgumentParser
from bfs import bfs_path_search
from liblgraph_python_api import Galaxy


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--db", type=str, default="./p1_db")
    parser.add_argument(
        "--graph_name", type=str, default="default", help="import graph name"
    )

    parser.add_argument(
        "--username", type=str, default="admin", help="database username"
    )
    parser.add_argument(
        "--password", type=str, default="73@TuGraph", help="database password"
    )

    return parser.parse_args()


def path_str(path):
    path = " -> ".join(map(str, path))
    return f"(src) {path} (dst)"


def main():
    args = parse_args()

    # initialize a Galaxy (grpah db) and open the graph
    galaxy = Galaxy(args.db)
    galaxy.SetCurrentUser(args.username, args.password)
    db = galaxy.OpenGraph(args.graph_name, False)

    # run processes on the graph
    print(path_str(bfs_path_search(db, src=0, dst=3)))
    print(path_str(bfs_path_search(db, src=0, dst=6)))
    print(path_str(bfs_path_search(db, src=0, dst=9)))

    # close the database and galaxy
    db.Close()
    galaxy.Close()


if __name__ == "__main__":
    main()
