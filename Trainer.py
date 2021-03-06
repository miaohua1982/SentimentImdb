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

class Trainer(object):
    def __init__(self, args):
        self._classifier = None
        self._optimizer = None
        self._args = args
        self._train_ds = None
        self._valid_ds = None
        self._test_ds = None

        self._decay_count = 0
        self._early_stop = False

    def sort_by_len(self, x_data, x_source_len, target_y):
        pos = x_source_len.detach().cpu().numpy().argsort()[::-1].tolist()
        return x_data[pos], x_source_len[pos], target_y[pos]
    
    def check_train_state(self, valid_loss):
        if len(valid_loss) < 2:
            return
        
        val_loss_last = valid_loss[-1]
        val_loss_lm = valid_loss[-2]
        if val_loss_last>val_loss_lm or abs(val_loss_last-val_loss_lm)<self._args.tolerate_err:
            self._decay_count += 1
            if self._decay_count >= self._args.decay_count:
                self._early_stop = True
        else:
            self._decay_count = 0
            self._early_stop = False

    def pre_train(self):
        # log
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to train', self._args.model, 'model')
        # set random
        set_seed_everywhere(self._args.seed, self._args.cuda)
        # load dataset
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to load imdb data')
        train_ds, valid_ds, test_ds = load_imdb_data_all(self._args.dataset_path)
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to load imdb data')
        # load pretrained weights
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to load pretrained weights')
        if self._args.tokenizer == 'bert':
            words2ind, pretrained_weights = load_bert_word2vec(self._args.bert_model)
        elif self._args.tokenizer != 'adbert':
            words2ind, pretrained_weights = load_glove_word2vec(self._args.glove_file_path)
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to load pretrained weights')
        
        # setup tokenizer
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to setup tokenizer')
        if self._args.tokenizer == 'spacy':
            tokenizer = SpacyTokenize('en_core_web_sm')
        elif self._args.tokenizer == 'bert':
            tokenizer = BertTokenize(self._args.bert_model)
        elif self._args.tokenizer == 'adbert':
            tokenizer = AdBertTokenize(self._args.bert_model)
        else:
            tokenizer = BaseTokenize()
        # to build vocab
        tokenizer.build_vocab(train_ds)
        with open(self._args.tokenizer_dump_path, 'wb') as vf:
            pickle.dump(tokenizer, vf)
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to setup tokenizer')

        # make word vector from pretrained weights
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to build wordvec from pretrained weights')
        if self._args.tokenizer != 'adbert':
            word2vec = make_word_embedding(words2ind, pretrained_weights, tokenizer._words_vocab)
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to make word embedding weights')
        # setup dataset
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to setup dataset')
        # #1# we use shorter length or the memory will be out of usage in attention computation
        # #2# the bert model has max sequence length is 512
        dataset = SentimentDataset(tokenizer, train_ds, valid_ds, test_ds, max_seq_len=self._args.seq_max_len)
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to setup dataset')
        # setup model
        if self._args.model == 'lstm':
            classifier = LstmClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, rnn_hidden_dim=self._args.rnn_hidden_dim, rnn_layers=self._args.rnn_layers, embedding_weights=word2vec, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind(),dropout=self._args.drop_out)
        elif self._args.model == 'fast':
            classifier = FastTextClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, embedding_weights=word2vec, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind())
        elif self._args.model == 'gru':
            classifier = GruClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, rnn_hidden_dim=self._args.rnn_hidden_dim, rnn_layers=self._args.rnn_layers, embedding_weights=word2vec, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind(),dropout=self._args.drop_out)
        elif self._args.model == 'transformer_enc':
            classifier = TransformerClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, ff_hidden=self._args.ff_hidden, max_seq_len=self._args.seq_max_len, heads=self._args.heads, enc_layers=self._args.enc_layers, embedding_weights=word2vec, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind(), dropout=self._args.drop_out)
        elif self._args.model == 'bert':
            classifier = BertClassifier(vocab_len=len(tokenizer._words_vocab), hidden_dim=self._args.ff_hidden, num_layers=self._args.rnn_layers, embedding_dim=self._args.embedding_size, embedding_weights=word2vec, classes_num=self._args.classes_num, padding_idx=tokenizer.get_pad_ind(), dropout=self._args.drop_out)
        elif self._args.model == 'adbert':
            classifier = AdBertClassifier(hidden_dim=self._args.ff_hidden, num_layers=self._args.rnn_layers, bert_model=self._args.bert_model, classes_num=self._args.classes_num, dropout=self._args.drop_out)
        else:
            classifier = CnnClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=self._args.embedding_size, classes_num=self._args.classes_num, out_channels=self._args.ff_hidden, embedding_weights=word2vec, filter_size=[3,4,5], padding_idx=tokenizer.get_pad_ind())

        # load classifier weights if needed
        if self._args.load_model:
            classifier.load_state_dict(torch.load(self._args.model_path))
        self._classifier = classifier.to('cuda:0' if self._args.cuda else 'cpu')
        self._optimizer = optim.Adam(classifier.parameters(), lr=self._args.learning_rate)#, weight_decay=0.00001)
        self._dataset = dataset
 
    def train(self):
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to train....................')
        try:
            valid_losses = []
            epoch_bar = tqdm(desc='training routine', total=self._args.num_epochs, position=0)
            for epoch_index in range(self._args.num_epochs):
                # train
                cur_train_loss, cur_train_acc = self.train_one_epoch(epoch_index)
                print('The train dataset has loss %.3f, acc %.3f' % (cur_train_loss, cur_train_acc))
                # validate
                cur_valid_loss, cur_valid_acc = self.valid_one_epoch(epoch_index)
                valid_losses.append(cur_valid_loss)
                print('The valid dataset has loss %.3f, acc %.3f' % (cur_valid_loss, cur_valid_acc))
                # update progress bar
                epoch_bar.update()
            
                # check train state
                self.check_train_state(valid_losses)
                # maybe update lr
                #scheduler.step(train_state['val_loss'][-1])
                # maybe early stop
                if self._early_stop:
                    break
        except KeyboardInterrupt:
            print("Exiting loop")
    
    def after_train(self):
        # save model
        torch.save(self._classifier.state_dict(), self._args.model_path)
        # do test
        test_loss, test_acc = self.test_one_epoch()
        print('The test dataset has loss %.3f, acc %.3f' % (test_loss, test_acc))
        
        # finish
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to train....................')

    def train_one_epoch(self, epoch_index):
        # Iterate over training dataset
        self._dataset.set_split('train')
        self._train_ds = DataLoader(dataset=self._dataset, batch_size=self._args.batch_size, shuffle=True, drop_last=True)
        # loss
        #loss_func = nn.BCEWithLogitsLoss()
        loss_func = nn.CrossEntropyLoss()
        # progress bar
        train_bar = tqdm(desc='split=train', total=len(self._train_ds), position=0)   
        
        running_loss = 0.0
        running_acc = 0.0
        self._classifier.train()
        train_bar.reset()
    
        device = 'cuda:0' if self._args.cuda else 'cpu'
        for batch_index, batch_data in enumerate(self._train_ds):
            # step 0. get data
            x_data, x_source_len, target_y = batch_data
            
            # setp 0.0 sort data by lens
            x_data, x_source_len, target_y = self.sort_by_len(x_data, x_source_len, target_y)
            
            # step 1. zero the gradients
            self._optimizer.zero_grad()

            # step 2. compute the output
            y_pred = self._classifier(x_data=x_data.to(device), x_len=x_source_len.to(device))
            y_pred = y_pred.squeeze()
            # step 3. compute the loss
            loss = loss_func(y_pred, target_y.to(device))
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use _ to take gradient step
            self._optimizer.step()
            
            # compute the accuracy
            acc_t = compute_accuracy_multi(y_pred, target_y.to(device))
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # update bar
            train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
            train_bar.update()

        return running_loss, running_acc

    def valid_one_epoch(self, epoch_index):
        # Iterate over validate dataset
        self._dataset.set_split('valid')
        self._valid_ds = DataLoader(dataset=self._dataset, batch_size=self._args.batch_size, shuffle=True, drop_last=True)
        # loss
        loss_func = nn.CrossEntropyLoss() 
        # val bar
        val_bar = tqdm(desc='split=val', total=len(self._valid_ds), position=0)

        running_loss = 0.
        running_acc = 0.
        self._classifier.eval()
        val_bar.reset()
        device = 'cuda:0' if self._args.cuda else 'cpu'
    
        for batch_index, batch_data in enumerate(self._valid_ds):
            # data
            x_data, x_source_len, target_y = batch_data

            # setp 0.0 sort data by lens
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
            
            val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
            val_bar.update()
            val_bar.refresh() # something may not update last position
            
        return running_loss, running_acc

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

            # setp 0.0 sort data by lens
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
            test_bar.refresh()  # something may not update last position

        return running_loss, running_acc
