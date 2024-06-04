# NOTE: You might need to change the source path according to your workdir
TUGRAPH_DIR=/root/tugraph-db/build/output
P3_DIR=/root/ai3602/p3_LinkPrediction

cd ${TUGRAPH_DIR}

# copy the data into /root/tugraph-db/build/outputs, where lgraph_import is located.
# NOTE: You might need to change the source path according to your workdir
cp -r ${P3_DIR}/p3_data/ ./

# use lgraph_import to import the graph
./lgraph_import -c ./p3_data/p3.conf --dir ./p3_db --graph default --overwrite 1

cd ${P3_DIR}
