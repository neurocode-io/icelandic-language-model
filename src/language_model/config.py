from os import environ
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass


env_path = Path(".")
load_dotenv()
print(f"Loading config from: {env_path.cwd()}")


@dataclass
class Settings:
    access_key = environ["ACCESS_KEY"]
