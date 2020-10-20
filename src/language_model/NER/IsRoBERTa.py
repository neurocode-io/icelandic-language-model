
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification
from pathlib import Path

from transformers import Trainer, TrainingArguments
from torch.utils.data.dataset import Dataset
from transformers.modeling_auto import AutoModelForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer
from language_model.lib import log, azure_storage

logger = log.get_logger(__file__)

# tokenizer = AutoTokenizer.from_pretrained("neurocode/IsRoBERTa")
# model = AutoModelForMaskedLM.from_pretrained("neurocode/IsRoBERTa")

class IsRoBERTa:
    def __init__(self, data_dir: Path, tokenizer: PreTrainedTokenizer, dataset: Dataset, local_rank=-1):
        assert data_dir, "data_dir input needed"

        self.model_dir = f"{data_dir}/results"
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.training_args = TrainingArguments(
            run_name=data_dir.name,
            local_rank=local_rank,
            num_train_epochs=10,
            output_dir=f"{self.model_dir}",
            overwrite_output_dir=False,
            per_device_train_batch_size=48,  # Nvidia K80 99%
            seed=42,
            gradient_accumulation_steps=4,
            save_total_limit=1,
        )

    def upload(self):
        paths = [str(x) for x in Path(self.model_dir).glob("*")]

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
            model = AutoModelForMaskedLM.from_pretrained("neurocode/IsRoBERTa")

        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer
        )

        trainer.train()

        trainer.save_model(f"{self.model_dir}")
        self.upload()
