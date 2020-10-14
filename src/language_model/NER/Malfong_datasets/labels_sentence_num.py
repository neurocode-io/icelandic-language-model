import pandas as pd 

def add_sentence_id(dataframe):
    ''''
    Input: dataframe with columns 'Word' and 'Tag'
    Output: pd.DataFrame() with an additional column 'Sentence', with sentence numbers correspondimg to 'Word'; a sentence
    ends with a '.' 
    '''
    dataframe['Sentence'] = ""
    id = 0
    i_list = [0]
    n_rows = len(dataframe)
    for i in range(n_rows):
        if dataframe['Word'].iloc[i] != ".":
            continue
        else: 
            i_list.append(i+1)
            til = i +1
            dataframe['Sentence'].iloc[i_list[id]: til] = id 
            id += 1
    return dataframe 


def get_labels(path_to_file, target_path):
    df = pd.read_csv(path_to_file, sep="\t", header=None)
    df.columns = ['Word', 'Tag']
    labels_np = df['Tag'].unique()
    pd.DataFrame(labels_np).to_csv(target_path + 'labels.txt', sep=",", header=None, index=False)
