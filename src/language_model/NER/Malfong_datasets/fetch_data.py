from malfong import fetch
from train_test_split import train_test_split
from labels_sentence_num import get_labels

# download data if it does not exist locally
path = '/home/elena/Projects/neurocode/icelandic-language-model/src/language_model/NER/Malfong_datasets/'
folder_name = 'MIM-GOLD-NER'
fetch(path, folder_name)

# split into train test txt files 
target_path = '/home/elena/Projects/neurocode/icelandic-language-model/src/language_model/NER/'
path_to_folder = path + folder_name
train_test_split(target_path, path_to_folder)

# get unique labels for train set
path_to_file = '/home/elena/Projects/neurocode/icelandic-language-model/src/language_model/NER/train_temp.txt'
get_labels(path_to_file, target_path)
