import requests
import gzip
from pathlib import Path
from language_model.lib.log import get_logger
from language_model.lib import azure_storage

logger = get_logger(__file__)


class Oscar:
    def __init__(self, data_dir: Path, file_name="oscar_data.txt"):
        assert data_dir, "data_dir input needed"

        self.data_dir = data_dir
        self.file_name = file_name

    def is_local_store(self):
        data = self.data_dir / self.file_name

        return data.exists()

    def download(self):
        if self.is_local_store():
            logger.info("Data already stored locally")
            return

        logger.info(f"Downloading to {self.data_dir}")

        url = "https://oscar-public.huma-num.fr/shuffled/is.txt.gz"
        target_path = self.data_dir / "oscar_data.gz"

        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise RuntimeError("Could not download file")

        with open(target_path, "wb") as f:
            f.write(response.raw.read())

        with gzip.open(target_path, "rb") as f:
            with open(self.data_dir / self.file_name, "wb") as o:
                o.write(f.read())

        Path(target_path).unlink()

    def upload(self):
        azure_storage.upload(self.data_dir / self.file_name)
