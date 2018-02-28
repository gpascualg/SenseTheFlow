FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER Guillem Pascual <gpascualg93@gmail.com>

# Update + dependencies #
#########################

RUN apt-get update && \
	apt-get install -y curl bzip2 software-properties-common zip g++ unzip cmake vim \
		libxrender1 libfontconfig1 git \
		swig pkg-config openjdk-8-jdk-headless autoconf locate build-essential \
                cuda-command-line-tools-9-0 cuda-cublas-dev-9-0 cuda-cudart-dev-9-0 \
		cuda-cufft-dev-9-0 cuda-curand-dev-9-0 cuda-cusolver-dev-9-0 \
		cuda-cusparse-dev-9-0 libcudnn7=7.0.5.15-1+cuda9.0 libcudnn7-dev=7.0.5.15-1+cuda9.0 \
		libpng12-dev libfreetype6-dev libzmq3-dev zlib1g-dev

# Get anaconda #
################
RUN curl -OL https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
	bash Anaconda3-5.0.1-Linux-x86_64.sh -b -p /opt/anaconda && \
	rm Anaconda3-5.0.1-Linux-x86_64.sh

## Export path
ENV PATH=/opt/anaconda/bin:/root/bin:/usr/local/bin:$PATH \
	LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

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
# 0.5.4 was working
RUN echo "startup --batch" >>/etc/bazel.bazelrc && \
        echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >>/etc/bazel.bazelrc && \
	curl -O -L https://github.com/bazelbuild/bazel/releases/download/0.8.0/bazel-0.8.0-installer-linux-x86_64.sh && \
	chmod +x bazel-0.8.0-installer-linux-x86_64.sh && \
	./bazel-0.8.0-installer-linux-x86_64.sh && \
	rm ./bazel-0.8.0-installer-linux-x86_64.sh


# Get tensorflow #
##################
RUN git clone --branch=r1.5 --depth=1 https://github.com/tensorflow/tensorflow
WORKDIR tensorflow

## Hack to make tensorflow build process use non-standard python location
RUN sed -i \
	-e "s/^#!\/usr\/bin\/env python$/#!\/opt\/anaconda\/bin\/python/" \
	third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl

## Setup bazel configuration variables
ENV PYTHON_BIN_PATH=/opt/anaconda/bin/python \
	USE_DEFAULT_PYTHON_LIB_PATH=1 \
	TF_NEED_MKL=1 \
	TF_DOWNLOAD_MKL=1 \ 
	TF_NEED_CUDA=1 \
	TF_NEED_OPENCL=0 \ 
	TF_NEED_JEMALLOC=1 \
	TF_NEED_HDFS=0 \
	TF_NEED_GDR=0 \
	TF_NEED_MPI=0 \
	TF_ENABLE_XLA=1 \
	TF_CUDA_CLANG=0 \
	TF_NEED_GCP=0 \
	TF_CUDA_VERSION=9.0 \
	TF_CUDNN_VERSION=7 \
        CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
	TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.1

RUN chmod +x configure && \
	sed -i -e '3,4d' configure && \
	./configure

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
	bazel build -s  --config=opt --config=cuda --verbose_failures --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --jobs=$(grep -c '^processor' /proc/cpuinfo) //tensorflow/tools/pip_package:build_pip_package

RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg && \
	pip install /tmp/tensorflow_pkg/$(ls /tmp/tensorflow_pkg)

## Back to root
WORKDIR ..



# Fetch RocksDB #
#################
RUN git clone https://github.com/facebook/rocksdb.git && \
	mkdir rocksdb/build
WORKDIR rocksdb/build
RUN git checkout v5.3.6 && \
	cmake .. && \
	make -j $(grep -c '^processor' /proc/cpuinfo) && make install
WORKDIR ../..
RUN rm -rf rocksdb



# Install other dependencies #
##############################
RUN pip install tqdm seaborn selenium pandas==0.19.2 keras


# Setup PYTHONPATH #
####################
ENV PYTHONPATH=/notebooks:/opt/python-libs


# Configure jupyter at startup
ENV LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
RUN echo "#!/bin/bash\n\
echo 'Generating config'\n\
jupyter-notebook --allow-root --generate-config --config=/etc/jupyter-notebook.py\n\
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
if [ -n \"\${FETCH_TF_CONTRIB}\" ] \n\
then \n\
    pip install git+https://www.github.com/farizrahman4u/keras-contrib.git \n\
    git clone https://github.com/gpascualg/SenseTheFlow.git /opt/python-libs/SenseTheFlow \n\
fi \n\
# SSH \n\
/usr/sbin/sshd \n\
# Start \n\
jupyter-notebook /notebooks --allow-root --config=/etc/jupyter-notebook.py &>/dev/null" > /opt/anaconda/run_jupyter.sh.tpl
RUN sed 's/ *$//' /opt/anaconda/run_jupyter.sh.tpl > /opt/anaconda/run_jupyter.sh
RUN chmod +x /opt/anaconda/run_jupyter.sh

# SSH #
#######

RUN apt-get install -y openssh-server && \
	mkdir /var/run/sshd && \
	mkdir -p  ~/.ssh && \
	chmod 700 ~/.ssh && \
	touch /root/.ssh/authorized_keys && \
	chmod 600 /root/.ssh/authorized_keys

EXPOSE 22

# Entry point #
###############
# Add Tini
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]
CMD ["/opt/anaconda/run_jupyter.sh"]
