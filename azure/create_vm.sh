#!/bin/bash

set -e
# Create a VM in northern Ireland via CLI:

resourceGroup=ne-icelandic-model-$(whoami)
vmName=ne-icelandic-model-$(whoami)

# neurocode.io
az account set -s e9a0397c-9b68-49ea-ae88-dcbd2f08e73e

az group create \
  -l northeurope \
  -n $resourceGroup

az vm create \
  -n $vmName \
  -g $resourceGroup \
  --image UbuntuLTS \
  --size Standard_NC6_Promo \
  --vnet-name ne-network-first-16 \
  --subnet ne-subnet-first-24 \
  --admin-username azureuser \
  --admin-password $VM_ADMIN_PASSWD \
  --authentication-type password


# Install docker:

az vm extension set \
  --resource-group $resourceGroup \
  --vm-name $vmName \
  --name customScript \
  --publisher Microsoft.Azure.Extensions \
  --settings ./azure/docker-ubuntu.json

# Install CUDA:

az vm extension set \
  --resource-group $resourceGroup \
  --vm-name $vmName \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute \
  --version 1.3

# Install docker nvidia:

az vm extension set \
  --resource-group $resourceGroup \
  --vm-name $vmName \
  --name customScript \
  --publisher Microsoft.Azure.Extensions \
  --settings ./azure/nvidia-docker.json

# Print the status:

az vm get-instance-view \
    --resource-group $resourceGroup \
    --name $vmName \
    --query "instanceView.extensions"
