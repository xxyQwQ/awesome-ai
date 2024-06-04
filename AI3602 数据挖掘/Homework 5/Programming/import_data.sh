# NOTE: You might need to change the source path according to your workdir
TUGRAPH_DIR=/root/tugraph-db/build/output
P2_DIR=/root/ai3602/p2_CommunityDetection

cd ${TUGRAPH_DIR}

# copy the data into /root/tugraph-db/build/outputs, where lgraph_import is located.
# NOTE: You might need to change the source path according to your workdir
cp -r ${P2_DIR}/p2_data/ ./

# use lgraph_import to import the graph
./lgraph_import -c ./p2_data/p2.conf --dir ./p2_db --graph default --overwrite 1

cd ${P2_DIR}
