import argparse 

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_path', type=str, help='train/val/test dataset')
    parser.add_argument('-g', '--glove_file_path',type=str, help='glove weights file path')
    parser.add_argument('-m', '--model_path', type=str, help='model file save&load path')
    parser.add_argument('-v', '--vect_file_path', type=str, help='vectorizer object file save&load path')
    parser.add_argument('-c', '--cuda', default=False, help='use gpu', action='store_true')
    parser.add_argument('-lr','--learning_rate', default=1e-3, type=float, help='set learning rate for model')
    parser.add_argument('-bs','--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=5, type=int, help='traning epochs')
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
