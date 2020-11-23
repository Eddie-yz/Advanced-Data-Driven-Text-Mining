import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from scipy.special import rel_entr
from sklearn.feature_extraction.text import TfidfVectorizer


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: 
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

def get_cate_keywords(train_docs, train_labels):
    train_sentences = [' '.join(words) for words in train_docs]
    vectorizer = TfidfVectorizer(stop_words='english', norm='l1', use_idf=True)
    vectorizer.fit(train_sentences)

    global_freq = vectorizer.transform([' '.join(train_sentences)]).toarray()[0]
    global_scores = dict()
    for word, idx in vectorizer.vocabulary_.items():
        global_scores[word] = global_freq[idx]

    cates_sentence= defaultdict(list)
    for sent, label in zip(train_sentences, train_labels):
        cates_sentence[label].append(sent)
    
    cate_keywords = defaultdict(list)
    for label, sents in cates_sentence.items():
        cate_freq = vectorizer.transform([' '.join(sents)]).toarray()[0]
        entropies = rel_entr(cate_freq, global_freq) # / np.log(2)
        word_scores = dict()
        for word, idx in vectorizer.vocabulary_.items():
            word_scores[word] = entropies[idx]
        scores = sorted(word_scores.items(), reverse=True, key=lambda x:x[1])
        for word, score in scores[:50]:
            cate_keywords[label].append(word)
    
    return cate_keywords

def embedding_from_pretrain(embedding_path):
    embedding_dict = dict()
    with open(embedding_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0].replace('<','').replace('>','')
            if word.isalpha() is False:
                continue
            vec = np.array(values[1:], dtype='float32')
            embedding_dict[word] = vec
    print (len(embedding_dict))
    return embedding_dict

def get_embeddings(word_list, embedding_path, embedding_dim=300):
    embedding_dict = embedding_from_pretrain(embedding_path)
    emb_matrix, word2ind = list(), dict()
    for idx, w in enumerate(word_list):
        emb_w = embedding_dict.get(w, None)
        if emb_w is None:
            emb_w = np.random.normal(scale=0.6, size=(embedding_dim, ))
        emb_matrix.append(emb_w)
        word2ind[w] = idx
    return np.array(emb_matrix), word2ind

def create_emb_layer(weights_matrix, trainable=True):
    weights_matrix = torch.from_numpy(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
