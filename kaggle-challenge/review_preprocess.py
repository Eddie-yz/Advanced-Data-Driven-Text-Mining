import os
import numpy as np
import pandas as pd
import pickle
from  core.clean_utils import text_clean
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

if __name__ == "__main__":
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

    data_path = 'Data'
    df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))

    df_train['text'] = df_train['review']
    df_test['text'] = df_test['review']

    train_docs = [text_clean(doc) for doc in df_train['text']]
    train_labels = [label2ind[name] for name in df_train['label']]
    train_docs, val_docs, train_labels, val_labels = train_test_split(train_docs, train_labels, test_size=0.1, random_state=1)
    test_docs = [text_clean(doc) for doc in df_test['text']]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train = [' '.join(doc) for doc in train_docs]
    X_val = [' '.join(doc) for doc in val_docs]
    X_test = [' '.join(doc) for doc in test_docs]
    X_train_emb = tokenizer(X_train, max_length=400, truncation=True, padding='max_length')
    X_val_emb = tokenizer(X_val, max_length=400, truncation=True, padding='max_length')
    X_test_emb = tokenizer(X_test, max_length=400, truncation=True, padding='max_length')

    save_file = dict()
    save_file['train_docs'] = train_docs
    save_file['train_labels'] = train_labels
    save_file['val_docs'] = val_docs
    save_file['val_labels'] = val_labels
    save_file['test_docs'] = test_docs
    save_file['ind2label'] = ind2label
    save_file['label2ind'] = label2ind
    pickle.dump(save_file, open(os.path.join('Data/processed', 'processed_data.pkl'), 'wb'))

    save_file = dict()
    save_file['X_train_emb'] = X_train_emb
    save_file['X_val_emb'] = X_val_emb
    save_file['X_test_emb'] = X_test_emb
    pickle.dump(save_file, open(os.path.join('Data/processed', 'processed_bert_emb.pkl'), 'wb'))