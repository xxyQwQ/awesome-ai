# dataset preparation

echo "[info] creating dataset folder"
cd dataset
mkdir {dev,eval,metadata}

echo "[info] preparing development set"
python collect.py "$1/audio/train/weak" "dev/wav.csv"
python extract.py "dev/wav.csv" "dev/feature.h5"
ln -s $(realpath "$1/label/train/weak.csv") "dev/label.csv"

echo "[info] preparing evaluation set"
python collect.py "$1/audio/eval" "eval/wav.csv"
python extract.py "eval/wav.csv" "eval/feature.h5"
ln -s $(realpath "$1/label/eval/eval.csv") "eval/label.csv"

echo "[info] preparing meta data"
cp "$1/label/class_label_indices.txt" "metadata/class_label_indices.txt"
cd ..
