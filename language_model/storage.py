import os
import gzip
import requests


class Storage:
    def __init__(self, client, data_dir=""):
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = f"{os.getcwd()}/data"

        print(self.data_dir)
        self.client = client

    def get_blob(self):
        for blob in self.client.list_blobs():
            if blob.name == "is.txt.gz":
                return blob

        return None

    def is_local_store(self):
        return os.path.exists(f"{self.data_dir}/is.txt")

    def download_from_internet(self):
        print(f"Downloading to {self.data_dir}")

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        url = "https://oscar-public.huma-num.fr/shuffled/is.txt.gz"
        target_path = f"{self.data_dir}/is.txt.gz"

        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise RuntimeError("Could not download file")

        with open(target_path, "wb") as f:
            f.write(response.raw.read())

        with open(target_path, "rb") as data:
            self.client.upload_blob(name="is.txt.gz", data=data)

    def download(self):
        blob = self.get_blob()

        if not blob:
            return self.download_from_internet()

        stream = self.client.download_blob(blob)
        target_path = f"{self.data_dir}/is.txt.gz"
        with open(target_path, "wb") as f:
            stream.readinto(f)

    def unzip(self):
        with gzip.open(f"{self.data_dir}/is.txt.gz", "rb") as compressed:
            with open("is.txt", "wb") as isl:
                isl.write(compressed.read())

    def fetch_dataset(self):
        if self.is_local_store():
            return

        self.download()
        self.unzip()
