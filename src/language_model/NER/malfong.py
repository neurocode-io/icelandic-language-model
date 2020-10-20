from zipfile import ZipFile
from pathlib import Path
import requests
import shutil
from language_model.lib import azure_storage

from language_model.lib.log import get_logger

logger = get_logger(__file__)


class Malfong:
    def __init__(self, data_dir: Path, file_name="malfong_ner.txt"):
        assert data_dir, "data_dir input needed"
        self.data_dir = data_dir
        self.file_name = file_name

    def is_local_store(self):
        data = self.data_dir / self.file_name

        return data.exists()

    def extract(self, zipfile_path):
        with ZipFile(zipfile_path, "r") as zf:
            for f in zf.namelist():
                if f.startswith('/'):
                    continue
                
                source = zf.open(f)
                target = open(self.data_dir / Path(f).name, "wb")

                with source, target:
                    shutil.copyfileobj(source, target)

        Path(zipfile_path).unlink()
        print("Data downloaded")

    def merge(self):
        files = [str(x) for x in Path(self.data_dir).glob("*.txt")]
        with open(self.data_dir / self.file_name, "w") as outfile:
            for file in files:
                with open(file) as infile:
                    outfile.write(infile.read())

    def upload(self):
        azure_storage.upload(self.data_dir / self.file_name)

    def download(self):
        if self.is_local_store():
            logger.info("Data already exists")
            return

        url = "http://www.malfong.is/tmp/3C786F63-4B3D-EEB0-61AA-552445E652BA.zip"
        target_path = self.data_dir / "malfong_ner.zip"

        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise RuntimeError("Could not download file")

        with open(target_path, "wb") as f:
            f.write(response.raw.read())

        self.extract(target_path)
        self.merge()
        self.upload()
