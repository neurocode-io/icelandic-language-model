from typing import List
from torch.utils.data import Dataset
from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer
from language_model.NER.types import InputFeatures

from language_model.NER import utils


class TokenClassificationDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length=128):
        examples = utils.read_examples_from_file(file_path)
        self.features = utils.convert_examples_to_features(
            examples=examples,
            label_list=utils.get_labels(),
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            cls_token_segment_id=0,
            pad_token_label_id=self.pad_token_label_id
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]
