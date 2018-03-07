#!/bin/bash

# Already installed
if [ -f /usr/bin/nvidia-container-runtime ]
then
	exit 0
fi

# Has nvidia-docker
if [ -f $(which nvidia-docker) ]
then
	docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
	apt-get purge -y nvidia-docker
fi

# Install nvidia runtimes
wget https://nvidia.github.io/libnvidia-container/ubuntu16.04/amd64/libnvidia-container1_1.0.0~alpha.3-1_amd64.deb
wget https://nvidia.github.io/libnvidia-container/ubuntu16.04/amd64/libnvidia-container-tools_1.0.0~alpha.3-1_amd64.deb
wget https://nvidia.github.io/nvidia-container-runtime/ubuntu16.04/amd64/nvidia-container-runtime_1.1.1+docker17.12.0-1_amd64.deb
dpkg -i libnvidia-container1_*.deb libnvidia-container-tools_*.deb nvidia-container-runtime_*.deb
rm libnvidia-container1_*.deb libnvidia-container-tools_*.deb nvidia-container-runtime_*.deb

# Configure docker daemon
cat > /etc/docker/daemon.json <<- EOM
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOM

# Fix for debian
DISTRO=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
DISTRO=${DISTRO:1:$(expr ${#DISTRO} - 2)}
DISTRO=${DISTRO,,}

if [ ${DISTRO:0:6} = "debian" ]
then
	sed -i 's/ldconfig.real/ldconfig/' /etc/nvidia-container-runtime/config.toml
fi

# Restart docker
service docker restart
