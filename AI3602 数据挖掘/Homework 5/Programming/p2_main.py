from argparse import ArgumentParser
from itertools import permutations
from tugraph_process import Process
from liblgraph_python_api import Galaxy


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--db", type=str, default="./p2_db")
    parser.add_argument(
        "--graph_name", type=str, default="default", help="import graph name"
    )

    parser.add_argument(
        "--username", type=str, default="admin", help="database username"
    )
    parser.add_argument(
        "--password", type=str, default="73@TuGraph", help="database password"
    )
    parser.add_argument(
        "--truth_path",
        type=str,
        default="/root/ai3602/p2_CommunityDetection/p2_data/label_reference.csv",
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default="/root/ai3602/p2_CommunityDetection/p2_data/p2_prediction.csv",
    )

    return parser.parse_args()


def main(args):
    TRUTH_PATH = args.truth_path
    OUTPUT_CSV_PATH = args.output_csv_path

    with open(TRUTH_PATH, "r", encoding="utf-8") as fi:
        _ = fi.readline()
        gt = [tuple(map(int, line.strip().split(","))) for line in fi.readlines()]

    gt_map = {x: y for x, y in gt}

    galaxy = Galaxy(args.db)
    galaxy.SetCurrentUser(args.username, args.password)
    db = galaxy.OpenGraph(args.graph_name)

    res = Process(db, gt_map)

    db.Close()
    galaxy.Close()

    if len(set(res.values())) != 5:
        print(f"#Communities != 5. Got {len(set(res.values()))}.")
    else:
        best_acc = 0.0
        best_reindexer = (0,)
        for reindexer in permutations(range(5)):
            acc = 0.0
            for idx, lbl in gt:
                if reindexer[res[idx]] == lbl:
                    acc += 1
            acc /= len(gt)
            if acc > best_acc:
                best_acc = acc
                best_reindexer = reindexer

        print(f"Best ACC: {best_acc}.")

        for x, y in res.items():
            res[x] = best_reindexer[y]

    with open(OUTPUT_CSV_PATH, "w") as fo:
        fo.write("id, category\n")
        for x, y in sorted(res.items(), key=lambda x: x[0]):
            fo.write(f"{x}, {y}\n")

    print(f"Result written to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
