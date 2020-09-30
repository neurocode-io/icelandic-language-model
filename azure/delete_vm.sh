#!/bin/bash

set -e


# Delete the previously created VM:

resourceGroup=ne-icelandic-model-$(whoami)
vmName=ne-icelandic-model-$(whoami)

# neurocode.io
az account set -s e9a0397c-9b68-49ea-ae88-dcbd2f08e73e

az vm delete \
  -n $vmName \
  -g $resourceGroup \
  --yes

az group delete \
  -n $resourceGroup \
  --yes