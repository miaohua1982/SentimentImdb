from tqdm import tqdm
import os
import time
import pickle
import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
import argparse
from LstmClassifier import LstmClassifier
from GruClassifier import GruClassifier
from CnnClassifier import CnnClassifier
from fastTextClassifier import FastTextClassifier
from TransformerClassifier import TransformerClassifier
from BertClassifier import BertClassifier, AdBertClassifier
from SentimentDataset import SentimentDataset

from Tokenize import BaseTokenize, SpacyTokenize, BertTokenize, AdBertTokenize
from Helper import load_glove_word2vec, load_bert_word2vec, load_imdb_data_all, sort_by_len, set_seed_everywhere, compute_accuracy, compute_accuracy_multi, make_word_embedding

class Tester(object):
    def __init__(self, args):
        self._classifier = None
        self._args = args
        self._test_ds = None

    def sort_by_len(self, x_data, x_source_len, target_y):
        pos = x_source_len.detach().cpu().numpy().argsort()[::-1].tolist()
        return x_data[pos], x_source_len[pos], target_y[pos]
    
    def pre_test(self):
        # log
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to test', self._args.model, 'model')    
        # set random
        set_seed_everywhere(self._args.seed, self._args.cuda)
        # setup dataset
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to load test dataset imdb')    
        test_ds = load_imdb_data_helper(self._args.dataset_path, 'test')
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to load test dataset imdb')
        # setup tokenizer object
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to load tokenizer object')
        with open(self._args.tokenizer_dump_path, 'rb') as vf:
            tokenizer = pickle.load(vf)
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to load vectorizer object')
        # setup dataset
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to setup dataset')
        dataset = SentimentDataset(tokenizer=tokenizer, test_ds=test_ds)
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to setup dataset')

        # setup model
        if self._args.model == 'lstm':
            classifier = LstmClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, rnn_hidden_dim=self._args.rnn_hidden_dim, rnn_layers=self._args.rnn_layers, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind(),dropout=self._args.drop_out)
        elif self._args.model == 'fast':
            classifier = FastTextClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind())
        elif self._args.model == 'gru':
            classifier = GruClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, rnn_hidden_dim=self._args.rnn_hidden_dim, rnn_layers=self._args.rnn_layers, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind(),dropout=self._args.drop_out)
        elif self._args.model == 'transformer_enc':
            classifier = TransformerClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, ff_hidden=self._args.ff_hidden, max_seq_len=self._args.seq_max_len, heads=self._args.heads, enc_layers=self._args.enc_layers, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind(), dropout=self._args.drop_out)
        elif self._args.model == 'bert':
            classifier = BertClassifier(vocab_len=len(tokenizer._words_vocab), hidden_dim=self._args.ff_hidden, num_layers=self._args.rnn_layers, embedding_dim=self._args.embedding_size, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind(), dropout=self._args.drop_out)
        elif self._args.model == 'adbert':
            classifier = AdBertClassifier(hidden_dim=self._args.ff_hidden, num_layers=self._args.rnn_layers, bert_model=self._args.bert_model, classes_num=self._args.classes_num, dropout=self._args.drop_out)
        else:
            classifier = CnnClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, classes_num=self._args.classes_num, out_channels=self._args.ff_hidden, filter_size=[3,4,5], padding_idx=tokenizer.get_pad_ind())

        # load classifier weights if needed
        classifier.load_state_dict(torch.load(self._args.model_path))
        self._classifier = classifier.to('cuda:0' if self._args.cuda else 'cpu')
        self._dataset = dataset
 
    def test(self):
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to test....................')
        try:
            test_loss, test_acc = self.test_one_epoch()
            print('The test dataset has loss %.3f, acc %.3f' % (test_loss, test_acc)) 
        except KeyboardInterrupt:
            print("Exiting test loop")
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to test....................')
    
    
    def test_one_epoch(self):
        # Iterate over validate dataset
        self._dataset.set_split('test')
        self._test_ds = DataLoader(dataset=self._dataset, batch_size=self._args.batch_size, shuffle=True, drop_last=True)
        # loss
        loss_func = nn.CrossEntropyLoss() 
        # test bar
        test_bar = tqdm(desc='split=test', total=len(self._test_ds), position=0)

        running_loss = 0.
        running_acc = 0.
        self._classifier.eval()
        test_bar.reset()
        device = 'cuda:0' if self._args.cuda else 'cpu'
        
        for batch_index, batch_data in enumerate(self._test_ds):
            # data
            x_data, x_source_len, target_y = batch_data
            x_data, x_source_len, target_y = self.sort_by_len(x_data, x_source_len, target_y)
            
            # compute the output
            y_pred = self._classifier(x_data=x_data.to(device), x_len=x_source_len.to(device))
            y_pred = y_pred.squeeze()
            # compute the loss
            loss = loss_func(y_pred, target_y.to(device))
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # compute the accuracy
            acc_t = compute_accuracy_multi(y_pred, target_y.to(device))
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        
            test_bar.set_postfix(loss=running_loss, acc=running_acc)
            test_bar.update()
            #test_bar.refresh()  # something may not update last position

        return running_loss, running_acc
