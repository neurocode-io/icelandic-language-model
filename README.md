# Language-Model

1 .Update Azure resource quota for CPUs / GPUs

Create a VM in northern Ireland via CLI:

resourceGroup=ne-icelandic-model
vmName=ne-icelandic-model


az vm create \
  -n $vmName \
  -g $resourceGroup \
  --image UbuntuLTS \
  --size Standard_NC6_Promo \
  --vnet-name ne-network-first-16 \
  --subnet ne-subnet-first-24 \
  --admin-username azureuser \
  --admin-password WlnBUTVNLBG5D9H9BvgaOVZ7S \
  --authentication-type password


Install docker:

az vm extension set \
  --resource-group $resourceGroup \
  --vm-name $vmName \
  --name customScript \
  --publisher Microsoft.Azure.Extensions \
  --settings ./azure/docker-ubuntu.json

Install CUDA:

az vm extension set \
  --resource-group $resourceGroup \
  --vm-name $vmName \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute \
  --version 1.3 

Install docker nvidia:

az vm extension set \
  --resource-group $resourceGroup \
  --vm-name $vmName \
  --name customScript \
  --publisher Microsoft.Azure.Extensions \
  --settings ./azure/nvidia-docker.json

Check status:

az vm get-instance-view \
    --resource-group $resourceGroup \
    --name $vmName \
    --query "instanceView.extensions"


Delete extension:

az vm extension delete \
    --resource-group m$resourceGroup \
    --vm-name $vmName \
    --name customScript



docker run --gpus all -it --rm --ipc=host -v /code:/code --name icelandic donchev7/icelandic-model:v0 bash