# NOTE: You might need to change the source path according to your workdir
TUGRAPH_DIR=/root/tugraph-db/build/output
P3_DIR=/root/ai3602/p3_LinkPrediction

cd ${TUGRAPH_DIR}

# check if p3_main.py exists, if not, create a symbolic link
if [ ! -f ./p3_main.py ]; then
    ln -s ${P3_DIR}/p3_main.py ./p3_main.py
fi

# run p3_main.py under TUGRAPH_DIR
python p3_main.py --lr 1e-3 --num_neg_samples 5

# return to the original directory
cd ${P3_DIR}
