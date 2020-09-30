# Icelandic language-model

## TODO
- [ ] Update Readme with .env setup
- [ ] Add python script for uploading model to azure storage
- [ ] Add docu how to run the container in the VM


docker run --gpus all -it --rm --ipc=host -v /tmp:/tmp --name icelandic donchev7/icelandic-model:v6b9c3f1 bash
