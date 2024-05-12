import csv
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('source', type=str, help='directory containing audio file')
parser.add_argument('target', type=str, help='path of output file')
args = parser.parse_args()

with open(args.target, 'w', newline='') as target:
    writer = csv.writer(target, delimiter='\t')
    writer.writerow(['audio_id', 'file_name'])
    for file in Path(args.source).iterdir():
        if file.suffix in ['.wav', '.flac']:
            writer.writerow([file.name, str(file.absolute())])
