import os
from pathlib import Path
from azure.storage.blob import ContainerClient
from azure.storage.blob import BlobClient


class Tokenizer:
    def __init__(self, client, tokenizer, data_dir=""):
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = f"{os.getcwd()}/data"

        print(self.data_dir)
        self.tokenizer = tokenizer
        self.client = client

    def is_local_store(self):
        return os.path.exists(f"{self.data_dir}/icelandic/vocab.json")

    def get_tokenizer_from_blob(self):
        for blob in self.client.list_blobs():
            if blob.name == "vocab.json" and blob.name == "merges.txt":
                return blob

        return None


    def train(self): 
        paths = [str(x) for x in Path(self.data_dir).glob("*.txt")]

        self.tokenizer.train(
            files=paths,
            vocab_size=52_000,
            min_frequency=2,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ],
        )

        Path(f"{self.data_dir}/icelandic").mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_model(f"{self.data_dir}/icelandic")
    
        with open (f"{self.data_dir}/icelandic", "rb") as tok:
            self.client.upload_blob(tok)



    def fetch_tokenizer(self):
        if  self.is_local_store():
            return 
        
        if self.get_tokenizer_from_blob():
            return 
        
        self.train()
