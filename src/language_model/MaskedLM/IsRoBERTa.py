from pathlib import Path

from transformers import RobertaConfig, DataCollatorForLanguageModeling, RobertaForMaskedLM, Trainer, TrainingArguments
from torch.utils.data.dataset import Dataset
from transformers.modeling_auto import AutoModelForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer
from language_model.lib import log, azure_storage

logger = log.get_logger(__file__)


class IsRoBERTa:
    def __init__(self, data_dir: Path, tokenizer: PreTrainedTokenizer, dataset: Dataset, local_rank=-1):
        assert data_dir, "data_dir input needed"

        self.model_dir = f"{data_dir}/results"
        self.dataset = dataset

        self.config = RobertaConfig(
            vocab_size=52_000,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        self.training_args = TrainingArguments(
            run_name=data_dir.name,
            local_rank=local_rank,
            learning_rate=0.00005,  # default 0.00005
            output_dir=f"{self.model_dir}",
            overwrite_output_dir=False,
            num_train_epochs=1,
            per_device_train_batch_size=48,  # Nvidia K80 99%
            seed=42,
            save_steps=10_000,
            save_total_limit=1,
        )

        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

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

            model = AutoModelForMaskedLM.from_pretrained(last_checkpoint, config=self.config)

        else:
            model = RobertaForMaskedLM(config=self.config)

        trainer = Trainer(
            model=model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.dataset,
            prediction_loss_only=True,
        )

        trainer.train()

        trainer.save_model(f"{self.model_dir}")
        self.upload()
