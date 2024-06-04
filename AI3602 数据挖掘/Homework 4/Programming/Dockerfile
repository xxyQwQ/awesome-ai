# Use the tugraph/tugraph-compile-centos7 image as base
FROM tugraph/tugraph-compile-centos7:1.3.0

# Goto home directory
WORKDIR /root

# Build TuGraph
RUN git clone --recursive https://github.com/TuGraph-family/tugraph-db.git && \
    cd /root/tugraph-db && \
    deps/build_deps.sh && \
    mkdir build && cd build && \
    cmake .. -DOURSYSTEM=centos -DENABLE_PREDOWNLOAD_DEPENDS_PACKAGE=1 && \
    make -j10 && \
    make package

# Install miniconda
ENV PATH="/opt/conda/bin:${PATH}"
RUN curl -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && conda init bash
