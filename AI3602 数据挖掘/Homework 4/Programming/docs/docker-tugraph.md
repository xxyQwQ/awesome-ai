# TuGraph Docker Container from Scratch

If possible, you should first consider using the provided docker image. However, if the docker image is not working properly, you can try following the instructions below and build your own image.

Please note that building such a image requires you to *compile* TuGraph from source. The compiling process requires a decent amount of memory and could take 15-30 minutes (depending on your machine). **Therefore, if you ever need to build this image on your own, consider using a machine with adequate computational resources, preferably a server.**

- You can either [compile an image from dockerfile](#1-compiling-an-image-from-dockerfile), or
- follow the instructions from TuGraph repository and [compile TuGraph from scratch](#2-compile-tugraph-from-scratch).

## 1. Compiling an Image from Dockerfile

A `Dockerfile` is provided in this assignment. It is exactly the Dockerfile the TA has used for composing `ybrua/ai3602-tugraph-centos7`. You can compose a new image using this dockerfile by

```sh
docker build -t ybrua/ai3602-tugraph-centos7:latest -t ybrua/ai3602-tugraph-centos7:1.0.0 .
```

**Note.** The build process includes compiling TuGraph. If the compilation terminates abruptly with errors, it is likely due to you have ran out of memory. You can consider reducing the `-j20` parameter in the Dockerfile

```dockerfile
# Line#8 to Line#14 in the Dockerfile
RUN git clone --recursive https://github.com/TuGraph-family/tugraph-db.git && \
    cd /root/tugraph-db && \
    deps/build_deps.sh && \
    mkdir build && cd build && \
    cmake .. -DOURSYSTEM=centos -DENABLE_PREDOWNLOAD_DEPENDS_PACKAGE=1 && \
    make -j20 && \  # j20 means using 20 threads for compiling. Reduce this number if you run out of memory.
    make package
```

Once the image is built successfully, you should be able to use it without trouble.

## 2. Compile TuGraph from Scratch

A less recommended way is to follow the instructions on the [official TuGraph repository](https://github.com/TuGraph-family/tugraph-db) and build TuGraph from scratch.

The first step is to pull the docker image for the compiled environment, provided by TuGraph

```sh
# pull the compile environment
docker pull tugraph/tugraph-compile-centos7:latest
```

Then run this docker. A sample code for running it is

```sh
docker run -it -d \
    -p 7070:7070 \
    -v C:/some/directory/for/this/container:/root \
    --name tugraph-test \
    tugraph/tugraph-compile-centos7:latest \
    /bin/bash
```

This will start the docker in detached mode, and you can connect to the docker by

```sh
docker exec -it tugraph-test bash
```

If everything goes smoothly, you should be in the docker container now. Then follow the steps below

```sh
git clone --recursive https://github.com/TuGraph-family/tugraph-db.git
cd tugraph-db
deps/build_deps.sh
mkdir build && cd build
cmake .. -DOURSYSTEM=centos -DENABLE_PREDOWNLOAD_DEPENDS_PACKAGE=1
make -j8  # 8 threads for compiling, change this if needed
make package
```

Once this is done, the TuGraph should be compiled successfully.
