import os
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from core.model_utils import CenterLoss
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


def cal_macro_f1(y_pre, y_gt):
    epi = 0.01
    confusion_mat = np.zeros((10,10))
    for pre, gt in zip(y_pre, y_gt):
        confusion_mat[int(pre), int(gt)] += 1
    f1_scores = list()
    for i in range(10):
        precision = confusion_mat[i,i] / (np.sum(confusion_mat[i, :]) + epi)
        recall = confusion_mat[i,i] / (np.sum(confusion_mat[:, i]) + epi)
        f1_scores.append(2 * precision * recall / (precision + recall + epi))
    return np.mean(f1_scores)


def validation(model, val_loader, device):
    model.eval()
    y_pre, y_gt = list(), list()
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        pre = np.argmax(outputs[0].data.cpu().numpy(), axis=1)
        y_pre.extend(pre)
        y_gt.extend(batch['labels'].data.numpy().flatten().tolist())
    print (confusion_matrix(y_gt, y_pre))
    return np.sum(np.array(y_pre) == np.array(y_gt))/len(y_pre), cal_macro_f1(y_pre, y_gt)

def test_bert(bert_model_name, test_loader, ind2label, device, output_name='predicted-review.csv'):
    model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=10)
    model.to(device)
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
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    dic_df.to_csv(os.path.join(output_path, output_name), index=False)

def train_bert(train_loader, val_loader, device, num_epoch=20, LR_Bert=1e-5, LR_CL=1e-2, alpha_CL=0.01, bert_model_name='bert-base-uncased', center_name=None):
    """
    LR_Bert: lr for bert model
    LR_CL: lr for center loss
    alpha_CL: weight for center loss. If alpha_CL == 0: train without center loss
    bert_model_name: pretrained bert model path
    center_name: pretrained center loss path

    """
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=10)
    model.to(device)
    CELoss = nn.CrossEntropyLoss()
    if alpha_CL > 0:
        CentLoss = CenterLoss(num_classes=10, feat_dim=model.config.hidden_size, use_gpu=(device!=torch.device('cpu')))
        if center_name is not None:
            CentLoss.load_state_dict(torch.load(center_name))
        params = list(model.parameters()) + list(CentLoss.parameters())
    else:
        params = list(model.parameters())
    optimizer = AdamW(params, lr=LR_Bert)
    best_f1 = 0.
    for epoch in tqdm.tqdm(range(num_epoch)):
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
            if alpha_CL > 0:
                loss_cent = CentLoss(base_outputs.pooler_output, labels.view(-1))
                loss = alpha_CL * loss_cent + loss_CE
            else:
                loss = loss_CE
            loss.backward()
            if alpha_CL > 0:
                for param in CentLoss.parameters():
                    # LR_CL is learning rate for center loss, e.g. LR_CL = 0.5
                    param.grad.data *= (LR_CL / (alpha_CL * LR_Bert))
            
            optimizer.step()
            epoch_CE_loss += loss_CE.data.cpu().numpy()
            if alpha_CL > 0:
                epoch_Cent_loss += loss_cent.data.cpu().numpy()
            else:
                epoch_Cent_loss = 0
            epoch_total_loss += loss.data.cpu().numpy()
            epoch_cnt += 1
        print ('training loss for epoch {}: CE loss: {} | Center loss: {} | Total loss: {}'.format(epoch, \
                                                                                                   epoch_CE_loss/epoch_cnt, \
                                                                                                   epoch_Cent_loss/epoch_cnt, \
                                                                                                   epoch_total_loss/epoch_cnt))
        if epoch_Cent_loss/epoch_cnt < 1:
            LR_CL = 0.5
        elif epoch_Cent_loss/epoch_cnt < 5:
            LR_CL = 1

        print ('start evaluating...')
        val_acc, val_f1 = validation(model, val_loader, device)
        if val_f1 > best_f1:
            model.eval()
            model.save_pretrained(os.path.join(model_path, 'best_bert_finetuned_model'))
            torch.save(CentLoss.state_dict(), os.path.join(model_path, 'best_center_model'))
            print ('Best epoch so far: {}'.format(epoch))
            best_f1 = val_f1
        print ('For epoch {}: val acc {} | val f1 {}'.format(epoch, val_acc, val_f1))
    
    model.eval()
    return model
