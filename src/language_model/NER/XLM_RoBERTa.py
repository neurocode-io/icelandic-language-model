from transformers import XLMRobertaForTokenClassification, AutoModelForTokenClassification, AutoConfig
from pathlib import Path

from transformers import Trainer, TrainingArguments
from torch.utils.data.dataset import Dataset
from language_model.NER.utils import get_labels
from language_model.lib import log, azure_storage

logger = log.get_logger(__file__)


class XLMRoBERTa:
    def __init__(self, data_dir: Path, dataset: Dataset, local_rank=-1):
        assert data_dir, "data_dir input needed"

        self.model_dir = f"{data_dir}/results"
        self.dataset = dataset
        self.config = AutoConfig.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(get_labels()),
            id2label={i: label for i, label in enumerate(get_labels())},
            label2id={label: i for i, label in enumerate(get_labels())},
        )
        self.training_args = TrainingArguments(
            run_name=data_dir.name,
            local_rank=local_rank,
            num_train_epochs=10,
            output_dir=f"{self.model_dir}",
            overwrite_output_dir=False,
            per_device_train_batch_size=48,
            seed=42,
            gradient_accumulation_steps=4,
            save_total_limit=1,
        )

    def upload(self):
        paths = [str(x) for x in Path(self.model_dir).glob("**/*")]

        for file in paths:
            azure_storage.upload(file)

    def has_started(self):
        paths = [str(x) for x in Path(self.model_dir).glob("*")]

        for path in paths:
            if "checkpoint" in path:
                return True

        return False

    def get_latest_checkpoint(self):
        paths = [str(x) for x in Path(self.model_dir).glob("*")]

        checkpoints = [path for path in paths if "checkpoint" in path]

        return sorted(checkpoints)[-1]

    def train(self):
        if self.has_started():
            last_checkpoint = self.get_latest_checkpoint()
            logger.info(f"Resuming training from: {last_checkpoint}")

            model = AutoModelForTokenClassification.from_pretrained(last_checkpoint, config=self.config)

        else:
            model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", config=self.config)

        trainer = Trainer(model=model, args=self.training_args, train_dataset=self.dataset)

        trainer.train()

        trainer.save_model(f"{self.model_dir}")
        self.upload()
