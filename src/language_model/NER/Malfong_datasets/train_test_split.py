from pathlib import Path 
import pandas as pd
import glob


def file_split(path_to_folder):
    ''' 
    Splits all files from the folder folder_name into train and test files.
    Input:
        path: path where the folder folder_name ist stored 
        folder_name: folder name containing all files to be split
    '''
    if Path(path_to_folder).exists == False:
        print('Folder with files does not exists / has a different name')
        return 

    list_files = glob.glob(path_to_folder +"/*.txt")
    split = int(len(list_files)*0.9) # 11 files will go to train, 2 for test
    # train and test files are paths! 
    train_files = list_files[:split]
    test_files = list_files[split:]
    return train_files, test_files

def read_textfiles(files, target_path, name):
        ''''
        Input: 
            files: a list with paths to files 
            target_path: where to save the new file with name 'name' 
        Concentate all train / test files into one train/test file, add sentence numbers and save as txt.files
        '''
        # creating dataframes 
        dfs = []
        for filename in files:
            dfs.append(pd.read_csv(filename, sep="\t", header=None))
    
        big_frame = pd.concat(dfs, ignore_index=True)
        big_frame.to_csv(target_path + name, sep="\t", header=False, index=False)

def train_test_split(target_path, path_to_folder):
    trainfile=Path(target_path+'train_temp.txt')
    testfile=Path(target_path+'test_temp.txt')
    if trainfile.exists() and testfile.exists():
        print('train and test files exist')
        exit()
    
    train_files, test_files = file_split(path_to_folder)
    read_textfiles(test_files, target_path, 'test_temp.txt')
    read_textfiles(train_files, target_path, 'train_temp.txt')






