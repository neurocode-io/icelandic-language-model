import os
import tarfile
import zipfile

os.chdir("/home/elena/Projects/neurocode/icelandic-language-model/src")
print(os.getcwd())

def extract_tar_files(tname):
    with tarfile.open(tname, "r:gz") as tf:
        tf.extractall("./data/")


# extract_tar_files("./data/isl_newscrawl_2019_300K.tar.gz")

def open_file(fname):
    in_f = open(fname, "r")
    out_f = open('./data/final_file.txt', "w")
    for line in in_f:
        items = line.split()
        # skipping row number
        new_items = items[1:]
        # putting text back together
        new_line = " ".join(new_items)
        out_f.write(new_line)
    in_f.close()
    out_f.close()

open_file("./data/isl_newscrawl_2019_300K/isl_newscrawl_2019_300K-sentences.txt")

