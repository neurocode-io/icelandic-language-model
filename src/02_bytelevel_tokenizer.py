import argparse
from tokenizers import ByteLevelBPETokenizer
from language_model.tokenizer import Tokenizer

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", type=str, help="Data directory for storage", default="")
argv = parser.parse_args()

data_dir = argv.data_dir

byteLevelTok = ByteLevelBPETokenizer()
T = Tokenizer(byteLevelTok, data_dir)

T.train()
