#!/bin/bash

set -e


export distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

# Because of Microsoft not properly installing cuda :/
ps -ef | grep apt | awk '{ print $2 }' | xargs kill -9
dpkg --configure -a


apt-get update
apt-get install -y nvidia-docker2
apt autoremove -y
systemctl restart docker


docker run --rm --gpus all --ipc=host nvidia/cuda:10.1-base nvidia-smi | grep "Driver Version"

if [ $? -eq 0 ]
then
  echo "All good"
else
  echo "nvidia docker install error"
  exit 1
fi
