import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from core.model_utils import CenterLoss
from core.train_utils import cal_macro_f1
from sklearn.metrics import confusion_matrix


class LSTMClassifer(nn.Module):
    def __init__(self, embedding_layer, nlayers, hidden_dim, num_labels):
        super().__init__()
        self.embeddings = embedding_layer
        self.vocab_size, self.embedding_dim = self.embeddings.weight.shape
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=nlayers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.sigmoid(self.fc(ht[-1]))


class Encoder(nn.Module):
    def __init__(self, embedding_layer, nlayers, hidden_dim, bidirectional):
        super().__init__()
        self.embeddings = embedding_layer
        self.vocab_size, self.embedding_dim = self.embeddings.weight.shape
        self.rnn = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=nlayers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        x (batch_size, seq_len): index of each word for sentences
        output (batch_size, seq_len, 2*hidden_dim): hidden states for all time steps at the last layer
        ht/ct (batch_size, 2*num_layer, hidden_dim): hidden/cell states for all layers at the last time step t
        """
        x = self.embeddings(x)
        x = self.dropout(x)
        output, (ht, ct) = self.rnn(x)
        return output, ht.permute(1,0,2), ct.permute(1,0,2)


class Attention2layerNNTanh(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention2layerNNTanh, self).__init__()

        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, all_hiddens):
        """
        all_hiddens (batch_size, seq_len, 2*hidden_dim): encoder outputs
        att_weights (batch_size, seq_len, 1): normalized attention weights for every time step
        """
        unnormalized_attention = self.fc2(self.act(self.fc1(all_hiddens)))
        att_weights = self.softmax(unnormalized_attention)
        return att_weights


class LSTMAttClassifier(nn.Module):
    def __init__(self, embedding_layer, enc_nlayers, hidden_dim, num_labels):
        super(LSTMAttClassifier, self).__init__()
        self.encoder = Encoder(embedding_layer, nlayers=enc_nlayers, hidden_dim=hidden_dim, bidirectional=True)
        self.attention = Attention2layerNNTanh(hidden_dim)
        self.fc = nn.Linear(2*hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        all_hiddens, ht, ct = self.encoder(x)
        att_weights = self.attention(all_hiddens)
        # weighted sum of hidden states from all time steps
        context = torch.sum(att_weights * all_hiddens, 1)
        return self.sigmoid(self.fc(context))


class ReviewClassifier(object):
    def __init__(self, emb_matrix, cate_keywords, word2ind, ind2label, use_attention=True, enc_nlayers=2, hidden_dim=30, num_labels=10, device=torch.device('cpu'), lstm_model_name=None, center_name=None):
        self.cate_keywords = cate_keywords
        self.word2ind = word2ind
        self.ind2label = ind2label
        self.use_attention = use_attention
        self.device = device
        self.num_labels = num_labels
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        self._init_lstm(emb_matrix, enc_nlayers, hidden_dim, lstm_model_name)
        self._init_center_loss(emb_matrix, center_name)
        self._prepare_kw_tensors_n_labels()

    def _init_lstm(self, emb_matrix, enc_nlayers, hidden_dim, lstm_model_name):
        emb_layer, _, _ = self.create_emb_layer(emb_matrix)
        emb_layer.to(self.device)
        if self.use_attention:
            self.model = LSTMAttClassifier(emb_layer, enc_nlayers=enc_nlayers, hidden_dim=hidden_dim, num_labels=self.num_labels)
        else:
            self.model = LSTMClassifer(emb_layer, nlayers=enc_nlayers, hidden_dim=hidden_dim, num_labels=self.num_labels)
        if lstm_model_name is not None:
            self.model.load_state_dict(torch.load(lstm_model_name))
        self.model.to(self.device)
    
    def _init_center_loss(self, emb_matrix, center_name):
        self.CentLoss = CenterLoss(num_classes=self.num_labels, feat_dim=emb_matrix.shape[1], use_gpu=(self.device!=torch.device('cpu')))
        if center_name is not None:
            self.CentLoss.load_state_dict(torch.load(center_name))
        self.CentLoss.to(self.device)
    
    def _prepare_kw_tensors_n_labels(self):
        kw_tensors, labels = list(), list()
        for label in range(self.num_labels):
            keywords = self.cate_keywords[label]
            kw_tensors.extend([self.word2ind[w] for w in keywords])
            labels.extend([label for w in keywords])
        self.kw_info = {'embeddings': torch.tensor(kw_tensors).to(self.device),
                        'labels': torch.tensor(labels).to(self.device),
                        }

    def create_emb_layer(self, weights_matrix, trainable=True):
        weights_matrix = torch.from_numpy(weights_matrix)
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if not trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, num_embeddings, embedding_dim
    
    def validation(self, val_loader):
        self.model.eval()
        y_pre, y_gt = list(), list()
        for X, y in val_loader:
            outputs = self.model(X.to(self.device))
            pre = np.argmax(outputs.data.cpu().numpy(), axis=1)
            y_pre.extend(pre)
            y_gt.extend(y.data.numpy().flatten().tolist())
        print (confusion_matrix(y_gt, y_pre))
        return np.sum(np.array(y_pre) == np.array(y_gt))/len(y_pre), cal_macro_f1(y_pre, y_gt)
    
    def evaluate(self, val_loader, epoch, best_f1):
        val_acc, val_f1 = self.validation(val_loader)
        if val_f1 > best_f1:
            self.model.eval()
            torch.save(self.model.state_dict(), os.path.join(self.model_path, 'kw_best_lstm'))
            torch.save(self.CentLoss.state_dict(), os.path.join(self.model_path, 'kw_best_center'))
            print ('Best epoch so far: {}'.format(epoch))
            best_f1 = val_f1
        print ('For epoch {}: val acc {} | val f1 {}'.format(epoch, val_acc, val_f1))
        return best_f1

    def train(self, train_loader, val_loader, lstm_lr=1e-4, cl_lr=0.1, alpha_cl=1e-2, num_epochs=20):
        CELoss = nn.CrossEntropyLoss()
        params = list(self.model.parameters()) + list(self.CentLoss.parameters())
        optimizer = optim.AdamW(params, lr=lstm_lr)
        best_f1 = 0.
        focus_cl = True
        for epoch in tqdm.tqdm(range(num_epochs)):
            epoch_CE_loss, epoch_Cent_loss, epoch_total_loss = 0., 0., 0.
            epoch_cnt = 0.
            self.model.train()
            if focus_cl:
                alpha_cl += 0.045
            for encoding, labels in tqdm.tqdm(train_loader):
                encoding = encoding.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(encoding)
                loss_CE = CELoss(output.view(-1, self.num_labels), labels.view(-1))
                if self.use_attention:
                    loss_cent = self.CentLoss(self.model.encoder.embeddings(self.kw_info['embeddings']), self.kw_info['labels'])
                else:
                    loss_cent = self.CentLoss(self.model.embeddings(self.kw_info['embeddings']), self.kw_info['labels'])
                loss = alpha_cl * loss_cent + loss_CE
                loss.backward()
                for param in self.CentLoss.parameters():
                    # cl_lr is learning rate for center loss, e.g. cl_lr = 0.5
                    param.grad.data *= (cl_lr / (alpha_cl * lstm_lr))
                
                optimizer.step()
                epoch_CE_loss += loss_CE.data.cpu().numpy()
                epoch_Cent_loss += loss_cent.data.cpu().numpy()
                epoch_total_loss += loss.data.cpu().numpy()
                epoch_cnt += 1
            print ('training loss for epoch {}: CE loss: {} | Center loss: {} | Total loss: {}'.format(epoch, \
                                                                                                    epoch_CE_loss/epoch_cnt, \
                                                                                                    epoch_Cent_loss/epoch_cnt, \
                                                                                                    epoch_total_loss/epoch_cnt))
            if epoch_Cent_loss/epoch_cnt < 0.8:
                focus_cl = False
                cl_lr = lstm_lr
                alpha_cl = 1e-2
            else:
                focus_cl = True
                cl_lr = 0.1

            print ('start evaluating...')
            best_f1 = self.evaluate(val_loader, epoch, best_f1)
    
    def test(self, test_loader):
        self.model.eval()
        y_pre = list()
        for X, _ in test_loader:
            outputs = self.model(X.to(self.device))
            pre = np.argmax(outputs.data.cpu().numpy(), axis=1)
            y_pre.extend(pre)
        dic = {"Id": list(), "Predicted": list()}
        for i, pred in enumerate(y_pre):
            dic["Id"].append(i)
            dic["Predicted"].append(self.ind2label[pred])
        
        dic_df = pd.DataFrame.from_dict(dic)
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
        dic_df.to_csv(os.path.join(output_path, 'predicted-lstm-center.csv'), index=False)