from zipfile import ZipFile
from pathlib import Path
import requests


class Malfoeng:
    def __init__(self, data_dir: Path, foulder_name="MIM-GOLD-NER"):
        assert data_dir, "data_dir input needed"

        self.data_dir = data_dir
        self.foulder_name = foulder_name

    def is_local_store(self):
        data = self.data_dir / self.foulder_name

        return data.exists()

    def fetch(self):
        if self.is_local_store():
            print('Data already stored locally')
            return 

        url = "http://www.malfong.is/tmp/3C786F63-4B3D-EEB0-61AA-552445E652BA.zip" 
        target_path = self.data_dir / "MIM-GOLD-NER.zip"

        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise RuntimeError("Could not download file")

        with open(target_path, "wb") as f:
            f.write(response.raw.read())

        zf = ZipFile(target_path, 'r')
        zf.extractall(self.data_dir / self.foulder_name)
        zf.close()

        Path(target_path).unlink()

