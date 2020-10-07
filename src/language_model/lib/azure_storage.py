from pathlib import Path
from azure.storage.blob import BlobServiceClient
from language_model.config import Settings

account_url = "https://neurocode2290877122.blob.core.windows.net"
storage_container = "icelandic-model"
creds = Settings().access_key

service = BlobServiceClient(account_url, creds)
client = service.get_container_client(storage_container)


def upload(local_path: Path):
    assert local_path, "local paths needed"

    remote_path = str(local_path)
    if remote_path.startswith('/'):
        remote_path = remote_path[1:]

    with open(local_path, "rb") as f:
        client.upload_blob(remote_path, f)


def download(remote_path: str, local_path: str):
    assert remote_path, "remote path needed"
    assert local_path, "local path needed"

    if remote_path.startswith("/"):
        remote_path = remote_path[1:]

    for blob in client.list_blobs():
        if blob.name.endswith(remote_path):
            stream = client.download_blob(blob)
            with open(local_path, "wb") as f:
                stream.readinto(f)


def exists(remote_path: str):
    assert remote_path, "remote paths needed"

    if remote_path.startswith("/"):
        remote_path = remote_path[1:]

    for blob in client.list_blobs():
        if blob.name.endswith(remote_path):
            return True

    return False