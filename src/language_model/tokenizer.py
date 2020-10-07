from pathlib import Path

from transformers.tokenization_roberta import RobertaTokenizerFast
from language_model.lib import (
    log,
    azure_storage
)
from transformers.tokenization_utils import PreTrainedTokenizer

logger = log.get_logger(__file__)

class Tokenizer:
    def __init__(self, data_dir: Path, tokenizer: PreTrainedTokenizer):
        assert data_dir, "data_dir input needed"

        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def is_local_store(self):
        vocab = self.data_dir / "vocab.json"
        merges = self.data_dir / "merges.txt"

        return vocab.exists() and merges.exists()

    def isComputed(self):
        if self.is_local_store():
            return True

        logger.info("Tokenizer not stored locally. Will check in azure")
    
        if azure_storage.exists(f"{self.data_dir}/vocab.json"):
            azure_storage.download(f"{self.data_dir}/vocab.json", f"{self.data_dir}/vocab.json")

        if azure_storage.exists(f"{self.data_dir}/merges.txt"):
            azure_storage.download(f"{self.data_dir}/merges.txt" , f"{self.data_dir}/merges.txt")
            return True
        
        return False


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

        self.tokenizer.save_model(f"{self.data_dir}")

    def create_tokenizer(self):
        if self.isComputed():
            logger.info("Tokenizer for this dataset has already been created")
            self.tokenizer = RobertaTokenizerFast.from_pretrained(f"{self.data_dir}", max_len=512)
            return

        logger.info(f"Training tokenizer on data in {self.data_dir}")

        self.train()
        azure_storage.upload(self.data_dir / "vocab.json")
        azure_storage.upload(self.data_dir / "merges.txt")
