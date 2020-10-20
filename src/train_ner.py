import argparse
import wandb
from pathlib import Path
from language_model.NER import utils
from language_model.NER import dataset
from language_model.NER.malfong import Malfong
from language_model.NER.IsRoBERTa import IsRoBERTa
from language_model.lib.log import get_logger
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", type=str, help="Data directory for storing the expirement", default="./data")
parser.add_argument(
    "--run_name", type=str, help="Name the model run . Defaults to icelandic-model", default="icelandic-model"
)
parser.add_argument(
    "--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.", default=-1
)

argv = parser.parse_args()

data_dir = argv.data_dir
run_name = argv.run_name
local_rank = argv.local_rank

logger = get_logger(__file__)

logger.info(f"Starting run: {run_name}")


wandb.login()

data_dir = Path(f"{data_dir}") / run_name
data_dir.mkdir(parents=True, exist_ok=True)
dataset_filename = "malfong_ner.txt"

malfong = Malfong(data_dir)
malfong.download()

data = utils.file_to_sentences(data_dir / dataset_filename)
max_seq_length = 128
tokenizer = AutoTokenizer.from_pretrained("neurocode/IsRoBERTa")
features = utils.convert_examples_to_features(data, tokenizer, utils.get_labels(), max_seq_length)
dataset = dataset.create_dataset(features)
model = IsRoBERTa(data_dir, tokenizer=tokenizer, dataset=dataset)



model.train()