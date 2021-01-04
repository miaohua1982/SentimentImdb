import torch
import random
import time
from tqdm import tqdm
from torch import optim, nn
from argparse import Namespace
from transformers import BertTokenizer
from torchtext import data
from torchtext import datasets
from BertClassifier import AdBertClassifier
from Helper import compute_accuracy_multi


class TorchTextTrainer(object):
    def __init__(self, args):
        self._args = args

    def pre_train(self):
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'setup bert tokenizer from pretrained', self._args.bert_model)
        tokenizer = BertTokenizer.from_pretrained(self._args.bert_model)
        def tokenize_and_cut(sentence):
            tokens = tokenizer.tokenize(sentence)
            tokens = tokens[:self._args.seq_max_len-2]
            return tokens
        TEXT = data.Field(batch_first=True, use_vocab=False, tokenize=tokenize_and_cut, preprocessing=tokenizer.convert_tokens_to_ids, init_token=tokenizer.cls_token_id, eos_token=tokenizer.sep_token_id, pad_token=tokenizer.pad_token_id, unk_token=tokenizer.unk_token_id)
        LABEL = data.LabelField(dtype=torch.long)
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'setup bert imdb dataset from torchtext datasets class')
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=self._args.dataset_path)
        train_data, valid_data = train_data.split(random_state=random.seed(self._args.seed))
        LABEL.build_vocab(train_data)
        print(time.strftime('%Y/%m/%d %H:%M:%S'), 'setup bert imdb dataset iterator')
        device = 'cuda:0' if self._args.cuda else 'cpu'
        self._train_iterator, self._valid_iterator, self._test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=self._args.batch_size, device=device)
        self._classifier = AdBertClassifier(hidden_dim=self._args.ff_hidden, num_layers=self._args.rnn_layers, bert_model=self._args.bert_model, classes_num=self._args.classes_num, dropout=self._args.drop_out)
        self._classifier.to(device)
    
    def train(self):
        print('Start to train adbert by texttorch.................................')
        optimizer = optim.Adam(self._classifier.parameters(), lr=self._args.learning_rate)#, weight_decay=0.00001)
        for i in range(self._args.num_epochs):
            train_loss, train_acc = self.train_one_epoch(optimizer, i)
            validate_loss, validate_acc = self.validate_one_epoch(i)
            print('%d epoch train loss:%.3f, acc:%.3f, val loss:%.3f, acc:%.3f' % (i, train_loss, train_acc, validate_loss, validate_acc))
            print('-'*60)

    def after_train(self):
        test_loss, test_acc = self.test_classifier()
        print('Train loss:%.3f, acc:%.3f' % (test_loss, test_acc))
        print('.'*60)

    def train_one_epoch(self, optimizer, epoch):
        device = 'cuda:0' if self._args.cuda else 'cpu'
        # Iterate over training dataset
        dataset = self._train_iterator
        # loss
        # loss_func = nn.BCEWithLogitsLoss()
        loss_func = nn.CrossEntropyLoss().to(device) 
        # progress bar
        train_bar = tqdm(desc='split=train', total=len(dataset), position=0)   
    
        running_loss = 0.0
        running_acc = 0.0
        self._classifier.train()
        train_bar.reset()
    
        for batch_index, batch_data in enumerate(dataset):
            # the training routine is these 6 steps:
            # step 0. get data
            x_data, target_y = batch_data.text, batch_data.label
            # step 1. zero the gradients
            optimizer.zero_grad()
            # step 2. compute the output
            y_pred = self._classifier(x_data=x_data)
            y_pred = y_pred.squeeze()
            # step 3. compute the loss
            loss = loss_func(y_pred, target_y)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # step 4. use loss to produce gradients
            loss.backward()
            # step 5. use optimizer to take gradient step
            optimizer.step()
            # -----------------------------------------
            # compute the accuracy
            acc_t = compute_accuracy_multi(y_pred, target_y.to(device))
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # update bar
            train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch)
            train_bar.update()

        return running_loss, running_acc

    def valid_one_epoch(self, epoch):
        device = 'cuda:0' if args.cuda else 'cpu'
        # Iterate over training dataset
        dataset = self._valid_iterator
        # loss
        #loss_func = nn.BCEWithLogitsLoss()
        loss_func = nn.CrossEntropyLoss().to(device) 
        # progress bar
        validate_bar = tqdm(desc='split=val', total=len(dataset), position=0)   

        running_loss = 0.0
        running_acc = 0.0
        classifier.eval()
        validate_bar.reset()

        for batch_index, batch_data in enumerate(dataset):
            # the validate routine is these 4 steps:
            # step 0. get data
            x_data, target_y = batch_data.text, batch_data.label
            # step 1. compute the output
            y_pred = classifier(x_data=x_data)
            y_pred = y_pred.squeeze()
            # step 2. compute the loss
            loss = loss_func(y_pred, target_y)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # setp 3. compute the accuracy
            acc_t = compute_accuracy_multi(y_pred, target_y.to(device))
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # update bar
            validate_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch)
            validate_bar.update()

        return running_loss, running_acc

    def test_classifier(self):
        device = 'cuda:0' if self._args.cuda else 'cpu'
        # Iterate over test dataset
        dataset = self._test_iterator
        # loss
        #loss_func = nn.BCEWithLogitsLoss()
        loss_func = nn.CrossEntropyLoss().to(device) 
        # progress bar
        test_bar = tqdm(desc='split=test', total=len(dataset), position=0)   
        
        running_loss = 0.0
        running_acc = 0.0
        self._classifier.eval()
        test_bar.reset()
        
        for batch_index, batch_data in enumerate(dataset):
            # the validate routine is these 4 steps:
            # step 0. get data
            x_data, target_y = batch_data.text, batch_data.label
            # step 1. compute the output
            y_pred = classifier(x_data=x_data)
            y_pred = y_pred.squeeze()
            # step 2. compute the loss
            loss = loss_func(y_pred, target_y)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # setp 3. compute the accuracy
            acc_t = compute_accuracy_multi(y_pred, target_y.to(device))
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # update bar
            test_bar.set_postfix(loss=running_loss, acc=running_acc)
            test_bar.update()

        return running_loss, running_acc