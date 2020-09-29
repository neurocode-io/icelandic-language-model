#!/bin/bash

distribution=ubuntu18.04
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-docker2

systemctl restart docker


docker run --rm --gpus all --ipc=host nvidia/cuda:10.1-base nvidia-smi | grep "Driver Version"

if [ $? -eq 0 ]
then
  echo "All good"
else
  echo "nvidia docker install error"
  exi 1
fi
