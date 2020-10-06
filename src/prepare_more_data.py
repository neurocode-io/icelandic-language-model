import os
import tarfile
import glob

# def extract_from_tar_files():
#     for tar_fname in glob.glob('/home/elena/Downloads/*.tar.gz'):
#         t=tarfile.open(tar_fname, 'r:gz')
#         t.extractall('.', members=[m for m in t.getmembers() if "sentences.txt" in m.name])
#         t.close()


# extract_from_tar_files()

def edit_file(fname):
    in_f = open(fname, "r")
    out_f = open('/home/elena/Projects/neurocode/icelandic-language-model/src/data/data_Leipzig.txt', "w")
    for line in in_f:
        items = line.split()
        # skipping row number
        new_items = items[1:]
        # putting text back together
        new_line = " ".join(new_items)
        out_f.write(new_line)
    in_f.close()
    out_f.close()

edit_file("/home/elena/Projects/neurocode/icelandic-language-model/src/data/isl_newscrawl_2019_300K/isl_newscrawl_2019_300K-sentences.txt")

