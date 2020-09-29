
from azure.storage.blob import BlobServiceClient
from language_model.storage import Storage
from language_model.config import settings


# TODO change to argparse
account_url = "https://neurocode2290877122.blob.core.windows.net"
storage_container = "icelandic-model"
creds = settings.access_key

service = BlobServiceClient(account_url, creds)
client = service.get_container_client(storage_container)


s = Storage(client)

s.fetch_dataset()
