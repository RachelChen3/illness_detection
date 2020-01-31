## =============================
# combine all data into one file
## =============================
import os
import glob
import pandas as pd
import re
import numpy as np

subdirectories = [x[0] for x in os.walk('./timesofindia')]
subdirectories.remove('./timesofindia')

col_names =  ['desp', 'label']
ill_df  = pd.DataFrame(columns = col_names)

for subdire in subdirectories:
    ill_name = subdire.split('/')[2]
    subfiles = glob.glob(subdire+ "/*.txt") # [x[0] for x in os.walk(subdire)]

    for files in subfiles:
        f = open(files, 'r')
        # split into sublist if there are two "\n"
        text_list = f.readlines()
        start = 0
        end = None
        for idx, item in enumerate(text_list):
            if item == "\n" and text_list[idx-1] == "\n" and idx!= len(text_list)-2:
                end = idx
                # remove citation [\d] and \n
                content = " ".join([re.sub(r"\[\d+\]", "", ele.replace("\n","") ) for ele in text_list[start:end]])
                if content:
                    ill_df.loc[len(ill_df)] = [content.strip().replace("@",""),ill_name]
                start = idx
                end = None

        # ill_df.loc[len(ill_df)] = [" ".join([ele.replace("\n","") for ele in f.readlines()]),ill_name]
        f.close()

# shuffle datset
ill_df = ill_df.sample(frac=1).reset_index(drop=True)

row_nums = ill_df.shape[0]
train_size = int(np.ceil(row_nums*0.7))
val_size = int(np.ceil(row_nums*0.2))
train_df, val_df, test_df = ill_df[:train_size],ill_df[train_size:train_size+val_size],ill_df[train_size+val_size:]
train_df.to_csv('data_train.csv',index = False, sep='@')
val_df.to_csv('data_val.csv',index = False,sep='@')
test_df.to_csv('data_test.csv',index = False, sep='@')
