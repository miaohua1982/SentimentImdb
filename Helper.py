import os
import re
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel                                                                                         

def load_glove_word2vec(glove_file_path):
    word2ind = {}
    word2vec = []
    counter = 0
    with open(glove_file_path) as f:
        lines = f.readlines()
        for one_line in lines:
            vec = one_line.split(' ')
            word2ind[vec[0]] = counter
            word2vec.append([float(one) for one in vec[1:]])
            counter += 1

    return word2ind, word2vec

def load_bert_word2vec(bert_model='bert-base-uncased'):
    bert = BertModel.from_pretrained(bert_model)
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    return tokenizer.vocab, bert.embeddings.word_embeddings.weight

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def load_imdb_data(imdb_path):
    flag_dir_name = os.path.basename(imdb_path)
    if flag_dir_name == 'neg':
        flag = 0
    else:
        flag = 1
    
    sentiments = []
    files = os.listdir(imdb_path)
    for one_file in files:
        path = os.path.join(imdb_path, one_file)
        with open(path) as f:
            line = f.read()
        #line = preprocess_text(line)
        sentiments.append(line)
    
    ds = {'sentiment':sentiments, 'flag':[flag]*len(sentiments)}
    ds = pd.DataFrame(ds)

    return ds

def load_imdb_data_helper(imdb_path, stage='train'):
    # train & valid ds
    imdb_neg_path = os.path.join(imdb_path, stage, 'neg')
    imdb_pos_path = os.path.join(imdb_path, stage, 'pos')
    neg_ds = load_imdb_data(imdb_neg_path)
    pos_ds = load_imdb_data(imdb_pos_path)

    all_ds = pd.concat([neg_ds, pos_ds], axis=0)
    all_ds = all_ds.sample(frac=1).reset_index(drop=True)  # shuffle

    return all_ds

def load_imdb_data_all(imdb_path, train_percent=0.9, val_percent=0.1):
    # train & valid ds
    all_ds = load_imdb_data_helper(imdb_path, 'train')
    ds_count = all_ds.shape[0]
    
    def split(row_id):
        if row_id < int(ds_count*train_percent):
            return 'train'
        else:
            return 'val'
    
    all_ds['split'] = all_ds.apply(lambda r: split(r.name), axis=1)
    train_ds = all_ds[all_ds.split=='train']
    valid_ds = all_ds[all_ds.split=='val']
    
    # test
    test_ds = load_imdb_data_helper(imdb_path, 'test')
    test_ds['split'] = 'test'

    return train_ds, valid_ds, test_ds
    

def make_word_embedding(words2ind, pretrained_word2vec, words_vocab):
    word2vec = torch.zeros(len(words_vocab), len(pretrained_word2vec[0]))

    for word, ind in words_vocab._voc_2_ind.items():
        if word in words2ind:
            vec = pretrained_word2vec[words2ind[word]]
            word2vec[ind] = torch.tensor(vec)
    return word2vec


def sort_by_len(x_data, x_source_len, target_y):
    pos = x_source_len.detach().cpu().numpy().argsort()[::-1].tolist()
    return x_data[pos], x_source_len[pos], target_y[pos]

def compute_accuracy(y_pred, target_y):
    y = torch.round(torch.sigmoid(y_pred))
    acc = torch.eq(y, target_y).float().mean().item()

    return acc

def compute_accuracy_multi(y_pred, target_y):
    y_pred_label = F.softmax(y_pred, dim=1).argmax(dim=1)
    acc = torch.eq(y_pred_label, target_y).float().mean().item()

    return acc

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

