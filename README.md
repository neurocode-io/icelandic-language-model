# Icelandic language-model

## TODO
- [ ] Update Readme with .env setup
- [ ] Add python script for uploading model to azure storage
- [ ] Add docu how to run the container in the VM
- [ ] Add package installation verifications (transformers / CUDA ...)

docker run --gpus all -it --rm --ipc=host -v /tmp:/tmp --name icelandic donchev7/icelandic-model:v1801a6c bash
