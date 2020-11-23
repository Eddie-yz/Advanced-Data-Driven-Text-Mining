import os
import numpy as np
import pandas as pd
from core.meta_utils import extract_names

def training_label_disambiguity(df_train, name_dict):
    for label, names in name_dict.items():
        inds = df_train['name'].apply(lambda text: text[2:-1].lower() in names)
        df_train.loc[inds, 'label'] = label
    df_train.to_csv(os.path.join('Data', 'train_processed.csv'))


if __name__ == '__main__':
    label2ind = {'american (new)': 0,
             'american (traditional)': 1,
             'asian fusion': 2,
             'canadian (new)': 3,
             'chinese': 4,
             'italian': 5,
             'japanese': 6,
             'mediterranean': 7,
             'mexican': 8,
             'thai': 9}

    ind2label = {ind: name for name, ind in label2ind.items()}
    df_train = pd.read_csv(os.path.join('Data', 'train.csv'))

    name_dict, _ = extract_names(df_train, label2ind, ind2label)
    training_label_disambiguity(df_train, name_dict)