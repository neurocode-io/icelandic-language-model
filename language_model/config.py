from os import environ
from pathlib import Path
from dotenv import load_dotenv
from language_model.utils import DictX

env_path = Path(".")
load_dotenv()
print(f"Loading config from: {env_path.cwd()}")

settings = DictX({
    "access_key": environ["ACCESS_KEY"],
})
