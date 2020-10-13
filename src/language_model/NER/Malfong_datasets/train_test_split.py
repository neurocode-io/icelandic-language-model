from pathlib import Path 
import pandas as pd
import glob


def add_sentence_id(data):
    ''''
    Input: dataframe with columns 'Word' and 'Tag'
    Output: pd.DataFrame() with an additional column 'Sentence', with sentence numbers correspondimg to 'Word'; a sentence
    ends with a '.' 
    '''
    data['Sentence'] = ""
    id = 0
    i_list = [0]
    n_rows = len(data)
    for i in range(n_rows):
        if data['Word'].iloc[i] != ".":
            continue
        else: 
            i_list.append(i+1)
            til = i +1
            data['Sentence'].iloc[i_list[id]: til] = id 
            id += 1
    return data 

def file_split(path, folder_name):
    ''' 
    Splits all files from the folder folder_name into train and test files.
    Input:
        path: path where the folder folder_name ist stored 
        folder_name: folder name containing all files to be split
    '''
    list_files = glob.glob(path + folder_name +"/*.txt")
    split = int(len(list_files)*0.9) # 11 files will go to train, 2 for test
    # train and test files are paths! 
    train_files = list_files[:split]
    test_files = list_files[split:]
    return train_files, test_files

def textfiles_and_tags(files, target_path, name):
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
        big_frame.columns = ['Word', 'Tag']

        # substract and save unique Tags
        unique_tags = big_frame['Tag'].unique()
        tags = pd.DataFrame(unique_tags)
        tags.to_csv(target_path + name + '_tags.txt', sep=",", header=None, index=False)

        # add sentence numbers and save
        #data = add_sentence_id(big_frame)
        big_frame.to_csv(target_path + name +'.txt', sep="\t", header=False, index=False)

def train_test_split(path, folder_name):
    trainfile=Path(path+'train_temp.txt')
    testfile=Path(path+'test_temp.txt')
    if trainfile.exists() and testfile.exists():
        print('train and test files exist')
        exit()
    
    train_files, test_files = file_split(path, folder_name)
    textfiles_and_tags(test_files, path, 'test_temp')
    textfiles_and_tags(train_files, path, 'train_temp')






