# NOTE: You might need to change the source path according to your workdir
TUGRAPH_DIR=/root/tugraph-db/build/output
P2_DIR=/root/ai3602/p2_CommunityDetection

cd ${TUGRAPH_DIR}

# check if p2_main.py exists, if not, create a symbolic link
if [ ! -f ./p2_main.py ]; then
    ln -s ${P2_DIR}/p2_main.py ./p2_main.py
fi

# run p2_main.py under TUGRAPH_DIR
python p2_main.py

# return to the original directory
cd ${P2_DIR}
