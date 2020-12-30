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

from Vectorizer import SentimentVectorizer

from Tokenize import BaseTokenize, SpacyTokenize, BertTokenize, AdBertTokenize
from Helper import load_glove_word2vec, load_bert_word2vec, load_imdb_data_all, sort_by_len, set_seed_everywhere, compute_accuracy, compute_accuracy_multi, update_train_state, make_train_state, make_word_embedding


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_path', type=str, help='train/val/test dataset')
    parser.add_argument('-g', '--glove_file_path',type=str, help='glove weights file path')
    parser.add_argument('-m', '--model_path', type=str, help='model file save&load path')
    parser.add_argument('-v', '--vect_file_path', type=str, help='vectorizer object file save&load path')
    parser.add_argument('-c', '--cuda', default=False, help='use gpu', action='store_true')
    parser.add_argument('-lr','--learning_rate', default=1e-3, type=float, help='set learning rate for model')
    parser.add_argument('-bs','--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=7, type=int, help='traning epochs')
    parser.add_argument('-es','--early_stopping_criteria', default=3, type=int, help='early stopping to torelant when acc is down')
    parser.add_argument('-em','--embedding_size', default=100, type=int, help='word embedding size')
    parser.add_argument('-rd','--rnn_hidden_dim', default=256, type=int, help='rnn(lstm, gnu...) hidden size')
    parser.add_argument('-rl','--rnn_layers', default=2, type=int, help='layers for rnn')
    parser.add_argument('-do','--drop_out', default=0.1, type=float, help='drop out for network')
    parser.add_argument('-te','--tolerate_err', default=1e-4, type=float, help='toerate err between 2 batch')
    parser.add_argument('-dc','--decay_count', default=3, type=float, help='decay count')
    parser.add_argument('-lm','--load_model', default=False, help='whether to load model', action='store_true')
    parser.add_argument('-tr','--train', default=False, help='set program to train mode', action='store_true')
    parser.add_argument('--test', default=False, help='set program to test mode', action='store_true')
    parser.add_argument('--model', default='lstm', type=str, help='choose a model to use(default is lstm)', choices=['lstm', 'gru', 'fast','cnn', 'bert', 'adbert', 'transformer_enc'])
    parser.add_argument('-t', '--tokenizer', default='spacy', type=str, help='choose a model to use(default is spacy)', choices=['split','spacy','bert', 'adbert'])
    parser.add_argument('--heads', default=4, type=int, help='number of multi heads for attention(default is 8)', choices=[4,8,16,64])
    parser.add_argument('-el','--enc_layers', default=6, type=int, help='number of encoder layer for transformer encoder structure(default is 6)', choices=[1,2,4,6,8])
    parser.add_argument('--ff_hidden', default=100, type=int, help='number of transformer structure feedforward network hidden layer dimention(default is 1024)', choices=[256, 512,768,1024,2048])
    parser.add_argument('-cn','--classes_num', default=2, type=int, help='number of classes(default is 2)')
    parser.add_argument('-ml','--seq_max_len', default=512, type=int, help='number of classes(default is 512)')
    parser.add_argument('-bm', '--bert_model', default='bert-base-uncased', type=str, help='pretrained bert model name(default is bert-base-uncased')

    return parser

running_args = Namespace(dataset_path="/home/miaohua/Documents/myfavor/pytorch-sentiment-analysis/.data/imdb/aclImdb/train",
                 glove_file_path="/home/miaohua/Documents/myfavor/pytorch-sentiment-analysis/.vector_cache/glove.6B.100d.txt",
                 model_path="/home/miaohua/Documents/myfavor/pytorch-sentiment-analysis/gpu/sentiment/model_storage/model_lstm.pth",
                 vect_file_path="/home/miaohua/Documents/myfavor/pytorch-sentiment-analysis/gpu/sentiment/model_storage/vect_file.dmp",
                 model='lstm',
                 tokenizer='spacy',
                 bert_model='bert-base-uncased',
                 cuda=False,
                 seed=1337,
                 learning_rate=1e-3,
                 batch_size=64,
                 num_epochs=7,
                 heads=8,
                 enc_layers=6,
                 ff_hidden=300,
                 seq_max_len=512,
                 classes_num=2,
                 early_stopping_criteria=5,
                 embedding_size=100,
                 rnn_hidden_dim=256,
                 rnn_layers=2,
                 drop_out=0.1,
                 tolerate_err=1e-4,
                 decay_count=3,
                 load_model=False,
                 train=False,
                 test=False)

def set_args(user_args, running_args):
    if user_args.dataset_path is not None:
        running_args.dataset_path = user_args.dataset_path   
 
    if user_args.glove_file_path is not None:
        running_args.glove_file_path = user_args.glove_file_path

    if user_args.model_path is not None:
        running_args.model_path = user_args.model_path

    if user_args.vect_file_path is not None:
        running_args.vect_file_path = user_args.vect_file_path

    running_args.heads = user_args.heads
    running_args.enc_layers = user_args.enc_layers
    running_args.ff_hidden = user_args.ff_hidden
    running_args.seq_max_len = user_args.seq_max_len
    running_args.classes_num = user_args.classes_num
    running_args.model = user_args.model
    running_args.tokenizer = user_args.tokenizer
    running_args.cuda = user_args.cuda
    running_args.learning_rate = user_args.learning_rate
    running_args.batch_size = user_args.batch_size
    running_args.num_epochs = user_args.epochs
    running_args.early_stopping_criteria = user_args.early_stopping_criteria
    running_args.embedding_size = user_args.embedding_size
    running_args.rnn_hidden_dim = user_args.rnn_hidden_dim
    running_args.rnn_layers = user_args.rnn_layers
    running_args.drop_out = user_args.drop_out
    running_args.tolerate_err = user_args.tolerate_err
    running_args.decay_count = user_args.decay_count
    running_args.load_model = user_args.load_model
    running_args.train = user_args.train
    running_args.test = user_args.test

    return running_args


def train_all_steps(classifier, dataset, args):
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)#, weight_decay=0.00001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)
        
    try:
        train_state = make_train_state()
        epoch_bar = tqdm(desc='training routine', total=args.num_epochs, position=0)

        for epoch_index in range(args.num_epochs):  
            train_state['epoch_index'] = epoch_index

            # train
            train_one_epoch(classifier, optimizer, dataset, args, train_state)

            # validate
            validate_one_epoch(classifier, dataset, args, train_state)
        
            # update progress bar
            epoch_bar.update()
            
            # check train state
            train_state = update_train_state(args=args, classifier=classifier, train_state=train_state)
            # maybe update lr
            #scheduler.step(train_state['val_loss'][-1])
            # maybe early stop
            if train_state['early_stop']:
                break
    except KeyboardInterrupt:
        print("Exiting loop")
    
    # save model
    torch.save(classifier.state_dict(), args.model_path)
    # do test
    test_one_epoch(classifier, dataset, args)

def train_one_epoch(classifier, optimizer, dataset, args, train_state):
    # Iterate over training dataset
    # loss
    #loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss() 
    # setup: batch generator, set loss and acc to 0, set train mode on
    dataset.set_split('train')
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # progress bar
    train_bar = tqdm(desc='split=train', total=dataset.get_num_batches(args.batch_size), position=0)   
    
    running_loss = 0.0
    running_acc = 0.0
    classifier.train()
    train_bar.reset()
    
    device = 'cuda:0' if args.cuda else 'cpu'
    for batch_index, batch_data in enumerate(dataloader):
        # the training routine is these 6 steps:
        #if batch_index >= 200:
        #    break
        # --------------------------------------
        # step 0. get data
        x_data, x_source_len, target_y = batch_data

        # setp 0.0 sort data by lens
        x_data, x_source_len, target_y = sort_by_len(x_data, x_source_len, target_y)
            
        # step 1. zero the gradients
        optimizer.zero_grad()

        # step 2. compute the output
        y_pred = classifier(x_data=x_data.to(device), x_len=x_source_len.to(device))
        y_pred = y_pred.squeeze()
        # step 3. compute the loss
        loss = loss_func(y_pred, target_y.to(device))
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
        train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=train_state['epoch_index'])
        train_bar.update()

    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)

def validate_one_epoch(classifier, dataset, args, train_state):
    # val bar
    # loss
    #loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss() 
    # Iterate over val dataset
    # setup: batch generator, set loss and acc to 0; set eval mode on
    dataset.set_split('val')
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # val bar
    val_bar = tqdm(desc='split=val', total=dataset.get_num_batches(args.batch_size), position=0)

    running_loss = 0.
    running_acc = 0.
    classifier.eval()
    val_bar.reset()
    device = 'cuda:0' if args.cuda else 'cpu'
    
    for batch_index, batch_data in enumerate(dataloader):
        # data
        x_data, x_source_len, target_y = batch_data

        # setp 0.0 sort data by lens
        x_data, x_source_len, target_y = sort_by_len(x_data, x_source_len, target_y)
            
        # compute the output
        y_pred = classifier(x_data=x_data.to(device), x_len=x_source_len.to(device))
        y_pred = y_pred.squeeze()
        # compute the loss
        loss = loss_func(y_pred, target_y.to(device))
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy_multi(y_pred, target_y.to(device))
        running_acc += (acc_t - running_acc) / (batch_index + 1)
            
        val_bar.set_postfix(loss=running_loss, acc=running_acc,  epoch=train_state['epoch_index'])
            
        val_bar.update()
        
        # something may not update last position
        val_bar.refresh()
    
    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)

def test_one_epoch(classifier, dataset, args):
    # test bar
    # loss
    #loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss() 
    # Iterate over test dataset
    # setup: batch generator, set loss and acc to 0; set test mode on
    dataset.set_split('test')
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test bar
    test_bar = tqdm(desc='split=test', total=dataset.get_num_batches(args.batch_size), position=0)

    running_loss = 0.
    running_acc = 0.
    classifier.eval()
    test_bar.reset()
    device = 'cuda:0' if args.cuda else 'cpu'
        
    for batch_index, batch_data in enumerate(dataloader):
        # data
        x_data, x_source_len, target_y = batch_data

        # setp 0.0 sort data by lens
        x_data, x_source_len, target_y = sort_by_len(x_data, x_source_len, target_y)
            
        # compute the output
        y_pred = classifier(x_data=x_data.to(device), x_len=x_source_len.to(device))
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
        
        # something may not update last position
        test_bar.refresh()
        
    print('The test dataset has loss %.3f, acc %.3f' % (running_loss, running_acc))

def test_all_steps(classifier, dataset, args):
    # log
    print('Start to do test on test dataset')
    # loss
    # loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss() 
    # Iterate over test dataset
    # setup: batch generator, set loss and acc to 0; set test mode on
    dataset.set_split('test')
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test bar
    test_bar = tqdm(desc='split=test', total=dataset.get_num_batches(args.batch_size), position=0)

    running_loss = 0.
    running_acc = 0.
    classifier.eval()
    test_bar.reset()
    device = 'cuda:0' if args.cuda else 'cpu'
        
    for batch_index, batch_data in enumerate(dataloader):
        # data
        x_data, x_source_len, target_y = batch_data

        # setp 0.0 sort data by lens
        x_data, x_source_len, target_y = sort_by_len(x_data, x_source_len, target_y)
            
        # compute the output
        y_pred = classifier(x_data=x_data.to(device), x_len=x_source_len.to(device))
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
        
        # something may not update last position
        test_bar.refresh()
        
    print('The test dataset has loss %.3f, acc %.3f' % (running_loss, running_acc))

def train(args):
    # log
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to train', args.model, 'model')
    # set random
    set_seed_everywhere(args.seed, args.cuda)
    
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to load imdb data')
    sentiment_ds = load_imdb_data_all(os.path.join(args.dataset_path,'neg'), os.path.join(args.dataset_path,'pos'))
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to load imdb data')
    
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to load glove weights')
    if args.tokenizer == 'bert':
        words2ind, pretrained_weights = load_bert_word2vec(args.bert_model)
    else:
        words2ind, pretrained_weights = load_glove_word2vec(args.glove_file_path)
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to load glove weights')
    
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to setup vectorizer')
    # setup tokenizer
    if args.tokenizer == 'spacy':
        tokenizer = SpacyTokenize('en_core_web_sm')
    elif args.tokenizer == 'bert':
        tokenizer = BertTokenize(args.bert_model)
    elif args.tokenizer == 'adbert':
        tokenizer = AdBertTokenize(args.bert_model)
    else:
        tokenizer = BaseTokenize()
    vectorizer = SentimentVectorizer.from_dataframe(sentiment_ds[sentiment_ds.split=='train'], tokenizer)
    with open(args.vect_file_path, 'wb') as vf:
        pickle.dump(vectorizer, vf)
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to setup vectorizer')
    
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to build wordvec from pretrained glove weights')
    word2vec = make_word_embedding(words2ind, pretrained_weights, tokenizer._words_vocab)
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to make word embedding weights')

    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to setup dataset')
    dataset = SentimentDataset(sentiment_ds, vectorizer)
    # #1# we use shorter length or the memory will be out of usage in attention computation
    # #2# the bert model has max sequence length is 512
    if args.model == 'transformer_enc' or args.model == 'adbert':
        dataset.set_max_len(args.seq_max_len)
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to setup dataset')
    
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to train....................')

    # setup model
    if args.model == 'lstm':
        classifier = LstmClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, \
                                 rnn_hidden_dim=args.rnn_hidden_dim, rnn_layers=args.rnn_layers, embedding_weights=word2vec, \
                                 classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind(),dropout=args.drop_out)
    
    elif args.model == 'fast':
        classifier = FastTextClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, embedding_weights=word2vec, \
                                        classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind())
   
    elif args.model == 'gru':
        classifier = GruClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, \
                                 rnn_hidden_dim=args.rnn_hidden_dim, rnn_layers=args.rnn_layers, embedding_weights=word2vec, \
                                 classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind(),dropout=args.drop_out)
    elif args.model == 'transformer_enc':
        classifier = TransformerClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, \
                                 ff_hidden=args.ff_hidden, max_seq_len=args.seq_max_len, heads=args.heads, enc_layers=args.enc_layers, embedding_weights=word2vec, \
                                 classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind(), dropout=args.drop_out)
    elif args.model == 'bert':
        classifier = BertClassifier(vocab_len=len(tokenizer._words_vocab), hidden_dim=args.ff_hidden, num_layers=args.rnn_layers, embedding_dim=args.embedding_size, embedding_weights=word2vec, \
                                    classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind(), dropout=args.drop_out)
    elif args.model == 'adbert':
        classifier = AdBertClassifier(hidden_dim=args.ff_hidden, num_layers=args.rnn_layers, bert_model=args.bert_model, \
                                    classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind(), dropout=args.drop_out)
    else:
        classifier = CnnClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, classes_num=args.classes_num, out_channels=args.ff_hidden,\
                                   embedding_weights=word2vec, filter_size=[3,4,5], padding_idx=tokenizer.get_mask_ind())

    # load classifier weights if needed
    if args.load_model:
        classifier.load_state_dict(torch.load(args.model_path))
    classifier = classifier.to('cuda:0' if args.cuda else 'cpu')
    train_all_steps(classifier, dataset, args)
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to train....................')


def test(args):
    # log
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to test', args.model, 'model')    
    # set random
    set_seed_everywhere(args.seed, args.cuda)

    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to load test dataset imdb')    
    sentiment_ds = load_imdb_data_all(os.path.join(args.dataset_path,'neg'), os.path.join(args.dataset_path,'pos'))
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to load test dataset imdb')

    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to load vectorizer object')
    with open(args.vect_file_path, 'rb') as vf:
        vectorizer = pickle.load(vf)
    tokenizer = vectorizer.get_tokenizer()
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to load vectorizer object')

    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to setup dataset')
    dataset = SentimentDataset(sentiment_ds, vectorizer)
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to setup dataset')
    
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Start to test............................')
    # setup model
    if args.model == 'lstm':
        classifier = LstmClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, \
                                    rnn_hidden_dim=args.rnn_hidden_dim, rnn_layers=args.rnn_layers, \
                                    classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind(),dropout=args.drop_out)
    
    elif args.model == 'fast':
        classifier = FastTextClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, \
                                        classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind())
   
    elif args.model == 'gru':
        classifier = GruClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, \
                                   rnn_hidden_dim=args.rnn_hidden_dim, rnn_layers=args.rnn_layers, \
                                   classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind(),dropout=args.drop_out)
    elif args.model == 'transformer_enc':
        classifier = TransformerClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, \
                                 ff_hidden=args.ff_hidden, max_seq_len=len(tokenizer._words_vocab), heads=args.heads, enc_layers=args.enc_layers, \
                                 classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind(), dropout=args.drop_out)
    elif args.model == 'bert':
        classifier = BertClassifier(vocab_len=len(tokenizer._words_vocab), hidden_dim=args.ff_hidden, num_layers=args.rnn_layers, embedding_dim=args.embedding_size,\
                                    classes_num=args.classes_num, padding_idx=tokenizer.get_mask_ind(), dropout=args.drop_out)
    else:
        classifier = CnnClassifier(vocab_len=len(tokenizer._words_vocab), embedding_dim=args.embedding_size, classes_num=args.classes_num, out_channels=args.ff_hidden,\
                                   filter_size=[3,4,5], padding_idx=tokenizer.get_mask_ind())

    # load classifier weights if needed
    classifier.load_state_dict(torch.load(args.model_path))
    classifier = classifier.to('cuda:0' if args.cuda else 'cpu')
    test_all_steps(classifier, dataset, args)
    print(time.strftime('%Y/%m/%d %H:%M:%S'), 'Finish to test....................')


if __name__ == '__main__':
    # args
    parser = args_parser()
    args = parser.parse_args()
    args = set_args(args, running_args)

    if args.train:
        print('exec training routine.......................................')
        train(args)
    
    if args.test:
        print('exec testing routine........................................')
        assert args.load_model==True, 'when testing, load_model must be True'
        test(args)


