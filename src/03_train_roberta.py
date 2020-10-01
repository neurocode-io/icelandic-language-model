import tokenizers
import argparse
import wandb
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from language_model.config import Settings

wandb.login()

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.", default=-1
)
parser.add_argument("--data_dir", type=str, help="Data directory for storage", default="./data")

argv = parser.parse_args()

local_rank = argv.local_rank
data_dir = argv.data_dir


tokenizers = RobertaTokenizerFast.from_pretrained(f"{data_dir}/icelandic", max_len=512)
model = RobertaForMaskedLM(config=config)

print(model.num_parameters())

dataset = LineByLineTextDataset(
    tokenizer=tokenizers,
    file_path=f"{data_dir}/is.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizers, mlm=True, mlm_probability=0.15)

print(f"local rank: {local_rank}")

training_args = TrainingArguments(
    local_rank=local_rank,
    output_dir=f"{data_dir}/icelandic",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    seed=42,
    save_steps=10_000,
    # save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()

trainer.save_model(f"{data_dir}/icelandic")


# python -m torch.distributed.launch --nnodes=2 --node_rank=0 --master_addr=12213 --master_port=1234 03_train_roberta.py --local_rank=0
