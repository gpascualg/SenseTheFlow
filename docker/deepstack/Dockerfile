#FROM ubuntu:16.04
FROM nvidia/cuda:latest-ubuntu16.04
MAINTAINER Guillem Pascual <gpascualg93@gmail.com>

# Update + dependencies #
#########################
RUN apt-get update && \
	apt-get install -y axel curl bzip2 software-properties-common git swig pkg-config zip g++ zlib1g-dev unzip \
		openjdk-8-jdk-headless libsnappy-dev libhdf5-serial-dev libatlas-base-dev liblmdb-dev cmake \
		autoconf protobuf-compiler && \
	apt-get install --no-install-recommends -y libboost-all-dev

# Get anaconda #
################
RUN curl -O -L https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
RUN bash Anaconda2-4.2.0-Linux-x86_64.sh -b -p /opt/anaconda

## Export path
ENV PATH=/opt/anaconda/bin:/root/bin:/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

## Configure anaconda
EXPOSE 8888

## Update conda
RUN conda install anaconda python=2.7 pip -y


# Get CUDNN 5 #
###############
ENV CUDNN_VERSION 5
LABEL com.nvidia.cudnn.version="5"
RUN CUDNN_DOWNLOAD_SUM=c10719b36f2dd6e9ddc63e3189affaa1a94d7d027e63b71c3f64d449ab0645ce && \
    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz -O && \
    echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v5.1.tgz" | sha256sum -c --strict - && \
    tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local && \
    rm cudnn-8.0-linux-x64-v5.1.tgz && \
    ldconfig


# Get Bazel #
#############
RUN curl -O -L https://github.com/bazelbuild/bazel/releases/download/0.4.4/bazel-0.4.4-installer-linux-x86_64.sh
RUN chmod +x bazel-0.4.4-installer-linux-x86_64.sh
RUN ./bazel-0.4.4-installer-linux-x86_64.sh --user


# Get tensorflow #
##################
RUN git clone https://github.com/tensorflow/tensorflow
WORKDIR tensorflow
RUN git checkout v1.0.0-rc2

## Hack to make tensorflow build process use non-standard python location
RUN sed -i \
	-e "s/^#!\/usr\/bin\/env python$/#!\/opt\/anaconda\/bin\/python/" \
	third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl

## Setup bazel configuration variables
ENV PYTHON_BIN_PATH=/opt/anaconda/bin/python
ENV TF_NEED_CUDA=1
ENV TD_NEED_OPENCL=0
ENV TF_NEED_JEMALLOC=1
ENV TF_NEED_HDFS=0
ENV TF_ENABLE_XLA=1
ENV TF_NEED_GCP=0
ENV TF_CUDA_VERSION=8.0
ENV TF_CUDNN_VERSION=5
RUN chmod +x configure
#RUN echo -e "#!/bin/bash\necho OOOOH\n$(cat configure)" > configure
RUN sed -i -e '3,4d' configure
RUN ./configure

## Build and install
RUN bazel build -s  -c opt --config=cuda --verbose_failures //tensorflow/tools/pip_package:build_pip_package
RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN pip install /tmp/tensorflow_pkg/tensorflow-1.0.0rc2-cp27-cp27mu-linux_x86_64.whl

## Back to root
WORKDIR ..

# Get Caffe dependencies #
##########################

## Opencv
RUN git clone https://github.com/opencv/opencv.git
RUN mkdir opencv/build
WORKDIR opencv/build
RUN cmake  -DCMAKE_BUILD_TYPE=RELEASE \
		-DCMAKE_INSTALL_PREFIX=/usr/local \
		-DBUILD_TIFF=ON -DPYTHON_EXECUTABLE=/opt/anaconda/bin/python \
		-DPYTHON_INCLUDE_DIRS=/opt/anaconda/include/python2.7 \
		-DPYTHON_LIBRARY=/opt/anaconda/lib/libpython2.7.so .. && \
	make -j8 install
WORKDIR ../..

## Google flags
RUN git clone https://github.com/gflags/gflags.git
RUN mkdir gflags/build
WORKDIR gflags/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON .. && make -j8 install
WORKDIR ../..

## Google log
RUN git clone https://github.com/google/glog.git
WORKDIR glog
RUN apt-get install -y libtool
RUN autoreconf --force --install && \
	./configure --prefix=/usr/local && \
	echo "#define GOOGLE_GLOG_DLL_DECL " >> src/config.h && \
	make -j8 install
WORKDIR ..

## Protobuf
RUN git clone https://github.com/google/protobuf.git
WORKDIR protobuf
RUN ./autogen.sh && ./configure --prefix=/usr/local && make -j8 install
WORKDIR ..

## Leveldb
RUN git clone https://github.com/google/leveldb.git
WORKDIR leveldb
RUN make -j8
RUN cp out-shared/*.so* /usr/local/lib/ && cp out-static/*.a* /usr/local/lib/ && cp -R include/* /usr/local/include
WORKDIR ..

## Caffe
RUN git clone https://github.com/BVLC/caffe.git
RUN mkdir caffe/build
WORKDIR caffe/build
RUN cmake .. && make -j8



# Install other dependencies #
##############################
RUN pip install tqdm
RUN conda install -y opencv
ENV PYTHONPATH=/notebooks

# Permanent volumnes #
######################
RUN mkdir /notebooks
VOLUME ["/notebooks"]

RUN mkdir /data
VOLUME ["/data"]

ENV PYTHONPATH=/caffe/python:$PYTHONPATH

RUN apt-get install -y libsm6 libxrender1 libfontconfig1
RUN echo $LD_LIBRARY_PATH
RUN apt-get install -y libsm6 libxrender1 libfontconfig1

# Configure jupyter at startup
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
RUN echo "#!/bin/bash\n\
echo 'Generating config'\n\
jupyter-notebook --generate-config --config=/etc/jupyter-notebook.py\n\
echo 'Replacing config with password'\n\
sed -i \ \n\
        -e \"s/^# *c.NotebookApp.ip = 'localhost'$/c.NotebookApp.ip = '0.0.0.0'/\" \ \n\
        -e \"s/^# *c.NotebookApp.port = 8888$/c.NotebookApp.port = 8888/\" \ \n\
        -e \"s/^# *c.NotebookApp.open_browser = True$/c.NotebookApp.open_browser = False/\" \ \n\
        -e \"s/^# *c.IPKernelApp.matplotlib = None$/c.IPKernelApp.matplotlib = 'inline'/\" \ \n\
        -e \"s/^# *c.NotebookApp.password = u''$/c.NotebookApp.password = u'\$JUPYTER_PASSWORD'/\" \ \n\
        -e \"s/^# *c.IPKernelApp.extensions = \[\]$/c.IPKernelApp.extensions = ['version_information']/\" \ \n\
        /etc/jupyter-notebook.py \n\
# Hackity hack to make anaconda behave
rm /opt/anaconda/lib/libstdc++.so.6 \n\
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/anaconda/lib/libstdc++.so.6 \n\
# Start \n\
jupyter-notebook /notebooks --config=/etc/jupyter-notebook.py > jupyter.log" > /opt/anaconda/run_jupyter.sh.tpl
RUN sed 's/ *$//' /opt/anaconda/run_jupyter.sh.tpl > /opt/anaconda/run_jupyter.sh
RUN chmod +x /opt/anaconda/run_jupyter.sh

## LAST TIME ADDITIONS
RUN pip install seaborn

# Entry point #
###############
ENTRYPOINT ["/opt/anaconda/run_jupyter.sh"]
