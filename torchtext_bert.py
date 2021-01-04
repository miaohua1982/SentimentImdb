import torch
import random
from tqdm import tqdm
from torch import optim, nn
from argparse import Namespace
from transformers import BertTokenizer
from torchtext import data
from torchtext import datasets
from BertClassifier import AdBertClassifier
from Helper import compute_accuracy_multi


args = Namespace(dataset_path="/home/miaohua/Documents/myfavor/pytorch-sentiment-analysis/.data",
                 model_path="./model_storage/model_torchtext_pre_bert.pth",
                 tokenizer='spacy',
                 bert_model='bert-base-uncased',
                 cuda=False,
                 seed=1337,
                 batch_size=128,
                 seq_max_len=512,
                 num_epochs=5,
                 classes_num=2,
                 ff_hidden=256,
                 rnn_layers=2,
                 drop_out=0.25,
                 learning_rate=0.001
                 )

def train_one_epoch(classifier, optimizer, dataset, args, epoch):
    device = 'cuda:0' if args.cuda else 'cpu'
    # Iterate over training dataset
    # loss
    #loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss().to(device) 
    # progress bar
    train_bar = tqdm(desc='split=train', total=len(dataset), position=0)   
    
    running_loss = 0.0
    running_acc = 0.0
    classifier.train()
    train_bar.reset()
    
    for batch_index, batch_data in enumerate(dataset):
        # the training routine is these 6 steps:
        # step 0. get data
        x_data, target_y = batch_data.text, batch_data.label
            
        # step 1. zero the gradients
        optimizer.zero_grad()

        # step 2. compute the output
        y_pred = classifier(x_data=x_data)
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

def validate_one_epoch(classifier, dataset, args, epoch):
    device = 'cuda:0' if args.cuda else 'cpu'
    # Iterate over training dataset
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

def test_classifier(classifier, dataset, args):
    device = 'cuda:0' if args.cuda else 'cpu'
    # Iterate over training dataset
    # loss
    #loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss().to(device) 
    # progress bar
    test_bar = tqdm(desc='split=test', total=len(dataset), position=0)   
    
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()
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


def train(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    def tokenize_and_cut(sentence):
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:args.seq_max_len-2]
        return tokens
    TEXT = data.Field(batch_first=True, use_vocab=False, tokenize=tokenize_and_cut, preprocessing=tokenizer.convert_tokens_to_ids, init_token=tokenizer.cls_token_id, eos_token=tokenizer.sep_token_id, pad_token=tokenizer.pad_token_id, unk_token=tokenizer.unk_token_id)

    LABEL = data.LabelField(dtype=torch.long)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=args.dataset_path)
    train_data, valid_data = train_data.split(random_state=random.seed(args.seed))

    LABEL.build_vocab(train_data)

    device = 'cuda:0' if args.cuda else 'cpu'
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=args.batch_size, device=device)

    classifier = AdBertClassifier(hidden_dim=args.ff_hidden, num_layers=args.rnn_layers, bert_model=args.bert_model, classes_num=args.classes_num, dropout=args.drop_out)
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)#, weight_decay=0.00001)
    
    for i in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(classifier, optimizer, train_iterator, args, i)
        validate_loss, validate_acc = validate_one_epoch(classifier, valid_iterator, args, i)
        print('-'*60)
        print('%d epoch train loss:%.3f, acc:%.3f, val loss:%.3f, acc:%.3f' % (i, 
        train_loss, train_acc, validate_loss, validate_acc))
        print('-'*60)

    test_loss, test_acc = test_classifier(classifier, test_iterator, args)
    print('-'*60)
    print('Train loss:%.3f, acc:%.3f' % (test_loss, test_acc))
    print('-'*60)


if __name__ == '__main__':
    train(args)