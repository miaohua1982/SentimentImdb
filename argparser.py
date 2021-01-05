import argparse 

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_path', type=str, help='train/val/test dataset')
    parser.add_argument('-g', '--glove_file_path',type=str, help='glove weights file path')
    parser.add_argument('-m', '--model_path', type=str, help='model file save&load path')
    parser.add_argument('-td', '--tokenizer_dump_path', type=str, help='tokenizer object file save&load path')
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
    parser.add_argument('-tt', '--torchtext', default=False, help='set program to train mode', action='store_true')

    return parser


def set_args(user_args, running_args):
    if user_args.dataset_path is not None:
        running_args.dataset_path = user_args.dataset_path   
 
    if user_args.glove_file_path is not None:
        running_args.glove_file_path = user_args.glove_file_path

    if user_args.model_path is not None:
        running_args.model_path = user_args.model_path

    if user_args.tokenizer_dump_path is not None:
        running_args.tokenizer_dump_path = user_args.tokenizer_dump_path

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
    running_args.torchtext = user_args.torchtext

    return running_args