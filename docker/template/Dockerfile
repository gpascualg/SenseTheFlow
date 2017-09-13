FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Guillem Pascual <gpascualg93@gmail.com>

# Update + dependencies #
#########################
RUN apt-get update && apt-get install -y curl bzip2 software-properties-common zip g++ unzip cmake vim \
		libxrender1 libfontconfig1 git

RUN apt-get install -y swig pkg-config openjdk-8-jdk-headless autoconf

# Get anaconda #
################
RUN curl -OL https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh 
RUN bash Anaconda3-4.3.1-Linux-x86_64.sh -b -p /opt/anaconda
RUN rm Anaconda3-4.3.1-Linux-x86_64.sh

## Export path
ENV PATH=/opt/anaconda/bin:/root/bin:/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

## Configure anaconda
EXPOSE 8888

# Update conda
RUN conda install anaconda python pip -y

# Permanent volumnes #
######################
RUN mkdir /notebooks
VOLUME ["/notebooks"]

RUN mkdir /data
VOLUME ["/data"]


# Get Bazel #
#############
RUN curl -O -L https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-installer-linux-x86_64.sh
RUN chmod +x bazel-0.4.5-installer-linux-x86_64.sh
RUN ./bazel-0.4.5-installer-linux-x86_64.sh --user


# Get tensorflow #
##################
RUN git clone https://github.com/tensorflow/tensorflow
WORKDIR tensorflow
RUN git checkout r1.2

## Hack to make tensorflow build process use non-standard python location
RUN sed -i \
	-e "s/^#!\/usr\/bin\/env python$/#!\/opt\/anaconda\/bin\/python/" \
	third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl

## Setup bazel configuration variables
RUN apt-get install -y locate build-essential
ENV PYTHON_BIN_PATH=/opt/anaconda/bin/python
ENV USE_DEFAULT_PYTHON_LIB_PATH=1
ENV TF_NEED_MKL=1
ENV TF_DOWNLOAD_MKL=1
ENV TF_NEED_CUDA=1
ENV TF_NEED_OPENCL=0
ENV TF_NEED_JEMALLOC=1
ENV TF_NEED_HDFS=0
ENV TF_ENABLE_XLA=1
ENV TF_CUDA_CLANG=0
ENV TF_NEED_GCP=0
ENV TF_CUDA_VERSION=8.0
ENV TF_CUDNN_VERSION=5
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.1
RUN chmod +x configure
RUN sed -i -e '3,4d' configure
RUN ./configure

## Build and install
RUN bazel build -s  --config=opt --config=cuda --verbose_failures //tensorflow/tools/pip_package:build_pip_package

RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN pip install /tmp/tensorflow_pkg/$(ls /tmp/tensorflow_pkg)

## Back to root
WORKDIR ..



# Fetch RocksDB #
#################
RUN git clone https://github.com/facebook/rocksdb.git
WORKDIR rocksdb
RUN git checkout v5.3.6
RUN mkdir build
WORKDIR build
RUN cmake ..
RUN make -j $(grep -c '^processor' /proc/cpuinfo) && make install
WORKDIR ../..




# Install other dependencies #
##############################
RUN pip install tqdm seaborn selenium


# Setup PYTHONPATH #
####################
ENV PYTHONPATH=/notebooks


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
        -e \"s/^# *c.NotebookApp.password = ''$/c.NotebookApp.password = '\$JUPYTER_PASSWORD'/\" \ \n\
        -e \"s/^# *c.IPKernelApp.extensions = \[\]$/c.IPKernelApp.extensions = ['version_information']/\" \ \n\
        /etc/jupyter-notebook.py \n\
# Hackity hack to make anaconda behave \n\
rm /opt/anaconda/lib/libstdc++.so.6 \n\
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/anaconda/lib/libstdc++.so.6 \n\
if [ -n \"\$ENABLE_GOMP_HACK\" ] \n\
then \n\
    rm /opt/anaconda/lib/libgomp.so.1 \n\
    ln -s $(find /usr/lib -name libgomp.so.1) /opt/anaconda/lib/libgomp.so.1 \n\
fi \n\
# Fetch latest SenseTheFlow \n\
if [ -n \"\${FETCH_SENSETHEFLOW}\" ] \n\
then \n\
    git clone https://github.com/gpascualg/SenseTheFlow.git /notebooks/SenseTheFlow \n\
fi \n\
# Start \n\
jupyter-notebook /notebooks --config=/etc/jupyter-notebook.py > jupyter.log" > /opt/anaconda/run_jupyter.sh.tpl
RUN sed 's/ *$//' /opt/anaconda/run_jupyter.sh.tpl > /opt/anaconda/run_jupyter.sh
RUN chmod +x /opt/anaconda/run_jupyter.sh


# Entry point #
###############
ENTRYPOINT ["/opt/anaconda/run_jupyter.sh"]
