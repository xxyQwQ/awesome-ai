import h5py
import librosa
import argparse
import pandas as pd
import pypeln as pl
import soundfile as sf
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('source', type=str, help='path of index file')
parser.add_argument('target', type=str, help='path of output file')
parser.add_argument('--num_workers', type=int, default=4, help='number of parallel workers')
parser.add_argument('--sample_rate', type=int, default=44100, help='sample rate of audio')
parser.add_argument('--hop_lenth', default=20, type=float, help='hop duration in ms')
parser.add_argument('--win_length', default=40, type=float, help='window duration in ms')
parser.add_argument('--num_bins', default=64, type=int, help='number of mel bins')
args = parser.parse_args()

audio_list = pd.read_csv(args.source, sep='\s+')
save_path = str(Path(args.target).absolute())
hop_length = int(args.sample_rate * args.hop_lenth / 1000)
win_length = int(args.sample_rate * args.win_length / 1000)

def extract(record):
    record = record[1]
    audio_wave, sample_rate = sf.read(record['file_name'], dtype='float32')
    if audio_wave.ndim > 1:
        audio_wave = audio_wave.mean(1)
    audio_wave = librosa.resample(audio_wave, orig_sr=sample_rate, target_sr=args.sample_rate)
    feature = librosa.feature.melspectrogram(y=audio_wave, sr=args.sample_rate, n_fft=2048, hop_length=hop_length, win_length=win_length, n_mels=args.num_bins)
    feature = librosa.power_to_db(feature, top_db=None).T
    return record['audio_id'], feature

target_data = []
with h5py.File(args.target, 'w') as store:
    for name, feature in pl.process.map(extract, audio_list.iterrows(), workers=args.num_workers, maxsize=4):
        store[name] = feature
        target_data.append({'audio_id': name, 'hdf5_path': save_path}) 
pd.DataFrame(target_data).to_csv(Path(args.target).with_suffix(".csv"), sep="\t", index=False)
