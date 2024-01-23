from .interface import launch
import argparse

args = argparse.ArgumentParser()
args.add_argument("--port", type=int, default=8111, help="Port to launch the GUI")
args = args.parse_args()

launch(args.port)