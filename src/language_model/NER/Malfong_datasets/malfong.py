from zipfile import ZipFile
from pathlib import Path
import requests


def is_local(path, folder_name):
    data = Path(path + folder_name)
    return data.exists()

def fetch(path, folder_name):
    if is_local(path, folder_name):
        print('Data already exists')
        return

    url = "http://www.malfong.is/tmp/3C786F63-4B3D-EEB0-61AA-552445E652BA.zip" 
    target_path = path+folder_name+'.zip'  

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError("Could not download file")

    with open(target_path, "wb") as f:
        f.write(response.raw.read())

    zf = ZipFile(target_path, 'r')
    zf.extractall(path)
    zf.close()
    
    Path(target_path).unlink()
    print('Data downloaded')