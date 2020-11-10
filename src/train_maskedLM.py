import argparse
import wandb
from pathlib import Path
from language_model.MaskedLM.IsRoBERTa import IsRoBERTa
from language_model.MaskedLM.dataset import Oscar
from language_model.lib.log import get_logger
from language_model.MaskedLM.tokenizer import Tokenizer
from tokenizers import ByteLevelBPETokenizer
from transformers import LineByLineTextDataset, TextDataset, RobertaTokenizerFast


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

dataset_filename = "oscar_data.txt"
oscar_data = Oscar(data_dir, file_name=dataset_filename)
oscar_data.download()

byteLevelTok = ByteLevelBPETokenizer()
tokenizer = Tokenizer(data_dir, byteLevelTok)
tokenizer.create_tokenizer()


tokenizers = RobertaTokenizerFast.from_pretrained(f"{data_dir}", max_len=512)

# LineByLineTextDataset, which splits your data into chunks, being careful not to overstep line returns as each line is interpreted as a document
#
# vs
#
# TextDataset, which just splits your data into chunks with no attention whatsoever to the line returns or separators
dataset = LineByLineTextDataset(
    tokenizer=tokenizers,
    file_path=f"{data_dir}/{dataset_filename}",
    block_size=128,
)

# https://arxiv.org/pdf/1907.11692.pdf
# e.g
# We pretrain with sequences of at most T = 512
# tokens. Unlike Devlin et al. (2019), we do not randomly inject short sequences, and we do not train
# with a reduced sequence length for the first 90% of
# updates. We train only with full-length sequences.
# dataset = TextDataset(tokenizer=tokenizers, file_path=f"{data_dir}/{dataset_filename}", block_size=512)

model = IsRoBERTa(data_dir, tokenizers, dataset, local_rank)
model.train()
