import argparse
from azure.storage.blob import BlobServiceClient
from language_model.storage import Storage
from language_model.config import Settings


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", type=str, help="Data directory for storage", default="")
argv = parser.parse_args()

data_dir = argv.data_dir

account_url = "https://neurocode2290877122.blob.core.windows.net"
storage_container = "icelandic-model"
creds = Settings().access_key

service = BlobServiceClient(account_url, creds)
client = service.get_container_client(storage_container)


s = Storage(client, data_dir)

s.fetch_dataset()
