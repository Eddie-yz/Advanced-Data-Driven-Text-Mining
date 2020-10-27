import pickle
import itertools
import os
import time
import pandas as pd
import numpy as np
import tqdm as tqdm
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import text_clean
from center_loss import CenterLoss

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def validation(model, val_loader, device):
    model.eval()
    y_pre, y_gt = list(), list()
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        pre = np.argmax(outputs[0].data.cpu().numpy(), axis=1)
        y_pre.extend(pre)
        y_gt.extend(list(batch['labels'].data.numpy()))
    return np.sum(y_pre == y_gt)/len(y_pre)

def train(train_loader, val_loader, device):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
    model.to(device)
    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in tqdm.tqdm(range(5)):
        epoch_loss = 0.
        epoch_cnt = 0.
        model.train()
        for batch in tqdm.tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            epoch_loss += loss.data.cpu().numpy()
            epoch_cnt += 1
            loss.backward()
            optim.step()
        print ('training loss for epoch {}: {}'.format(epoch, epoch_loss/epoch_cnt))
        print ('start evaluating...')
        val_acc = validation(model, val_loader, device)
        model.eval()
        model.save_pretrained(os.path.join('models', 'bert_finetuned_models_epoch{}'.format(epoch)))
        print ('val acc for epoch {}: {}'.format(epoch, val_acc))
    
    model.eval()
    return model

def train_with_center_loss(train_loader, val_loader, device, alpha_CL=0.01, LR_CL=1e3):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
    # model = BertForSequenceClassification.from_pretrained(os.path.join('models', 'bert_finetuned_models_epoch4'), num_labels=10)
    model.to(device)
    CELoss = nn.CrossEntropyLoss()
    CentLoss = CenterLoss(num_classes=10, feat_dim=model.config.hidden_size, use_gpu=(device!=torch.device('cpu')))
    params = list(model.parameters()) + list(CentLoss.parameters())
    lr = 5e-5
    optimizer = AdamW(params, lr=lr)

    for epoch in tqdm.tqdm(range(15)):
        epoch_CE_loss, epoch_Cent_loss, epoch_total_loss = 0., 0., 0.
        epoch_cnt = 0.
        model.train()
        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            base_outputs = model.bert(input_ids, attention_mask=attention_mask, output_hidden_states=False, return_dict=True)
            pooled_output = model.dropout(base_outputs.pooler_output)
            logits = model.classifier(pooled_output)
            loss_CE = CELoss(logits.view(-1, model.num_labels), labels.view(-1))
            loss_cent = CentLoss(base_outputs.pooler_output, labels.view(-1))
            loss = alpha_CL * loss_cent + loss_CE
            loss.backward()
            for param in CentLoss.parameters():
                # LR_CL is learning rate for center loss, e.g. LR_CL = 0.5
                param.grad.data *= (LR_CL / (alpha_CL * lr))
            
            optimizer.step()
            epoch_CE_loss += loss_CE.data.cpu().numpy()
            epoch_Cent_loss += loss_cent.data.cpu().numpy()
            epoch_total_loss += loss.data.cpu().numpy()
            epoch_cnt += 1
        print ('training loss for epoch {}: CE loss: {} | Center loss: {} | Total loss: {}'.format(epoch, \
                                                                                                   epoch_CE_loss/epoch_cnt, \
                                                                                                   epoch_Cent_loss/epoch_cnt, \
                                                                                                   epoch_total_loss/epoch_cnt))
        print ('start evaluating...')
        val_acc = validation(model, val_loader, device)
        model.eval()
        model.save_pretrained(os.path.join('models', 'bert_center_alpha1e2_lr1e3_epoch{}'.format(epoch)))
        print ('val acc for epoch {}: {}'.format(epoch, val_acc))
    
    model.eval()
    return model

def test(model, test_loader, ind2label, device):
    y_pre = list()
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        pre = np.argmax(outputs[0].data.cpu().numpy(), axis=1)
        y_pre.extend(pre)
    dic = {"Id": [], "Predicted": []}
    for i, pred in enumerate(y_pre):
        dic["Id"].append(i)
        dic["Predicted"].append(ind2label[pred])
    
    dic_df = pd.DataFrame.from_dict(dic)
    dic_df.to_csv(os.path.join('outputs', 'predicted-bert-center.csv'), index=False)

def main():
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
    train_docs, val_docs, train_labels, val_labels = train_test_split(train_docs, train_labels, test_size=0.2, random_state=1)
    test_docs = [text_clean(doc) for doc in df_test['text']]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train = [' '.join(doc) for doc in train_docs]
    X_val = [' '.join(doc) for doc in val_docs]
    X_test = [' '.join(doc) for doc in test_docs]
    X_train_emb = tokenizer(X_train, max_length=400, truncation=True, padding='max_length')
    X_val_emb = tokenizer(X_val, max_length=400, truncation=True, padding='max_length')
    X_test_emb = tokenizer(X_test, max_length=400, truncation=True, padding='max_length')

    train_dataset = TextDataset(X_train_emb, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = TextDataset(X_val_emb, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataset = TextDataset(X_test_emb, np.zeros(len(X_test_emb['input_ids'])))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = train(train_loader, val_loader, device)
    test(model, test_loader, ind2label, device)

if __name__ == '__main__':
    main()

