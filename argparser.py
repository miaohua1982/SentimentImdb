import argparse 

parser = argparse.ArgumentParser() 

parser.add_argument('-a', type=int, default=100, help='input a int')
parser.add_argument('-d', '--dataset_path', type=str, help='train/val/test dataset')
parser.add_argument('-g', '--glove_file_path',type=str, help='glove weights file path')
parser.add_argument('-m', '--model_path', type=str, help='model file save&load path')
parser.add_argument('-c', '--cuda', default=False, help='use gpu', action='store_true')
parser.add_argument('-lr','--learning_rate', default=1e-3, type=float, help='set learning rate for model')
parser.add_argument('-bs','--batch_size', default=64, type=int, help='batch size')
parser.add_argument('-e', '--epochs', default=7, type=int, help='traning epochs')
parser.add_argument('-es','--early_stopping_criteria', default=3, type=int, help='early stopping to torelant when acc is down')
parser.add_argument('-em','--embedding_size', default=100, type=int, help='word embedding size')
parser.add_argument('-rd','--rnn_hidden_dim', default=256, type=int, help='rnn(lstm, gnu...) hidden size')
parser.add_argument('-rl','--rnn_layers', default=2, type=int, help='layers for rnn')
parser.add_argument('-do','--drop_out', default=0.5, type=float, help='drop out for network')
parser.add_argument('-t', '--tolerate_err', default=1e-4, type=float, help='toerate err between 2 batch')
parser.add_argument('-dc','--decay_count', default=3, type=float, help='decay count')
parser.add_argument('-lm','--load_model', default=False, help='whether to load model', action='store_true')


args = parser.parse_args()

#                 dataset_path="../../.data/imdb/aclImdb/train",
#                 glove_file_path="../../.vector_cache/glove.6B.100d.txt",
#                 model_save_path="model_storage/model.pth",
#                 cuda=False,
#                 seed=1337,
#                 learning_rate=1e-3,
#                 batch_size=64,
#                 num_epochs=7,
#                 early_stopping_criteria=5,
#                 embedding_size=100,
#                 rnn_hidden_dim=256,
#                 rnn_layers=2,
#                 drop_out=0.5,
#                 tolerate_err=1e-4,
#                 decay_count=3,
#                 load_model=False


print(args.a)
#print(args.g)
print(args.glove_file_path)
print(args.cuda)
