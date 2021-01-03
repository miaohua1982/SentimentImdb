from argparser import set_args, args_parser
from Trainer import Trainer
from Tester import Tester

running_args = Namespace(dataset_path="/Users/miaohua/Desktop/myfavor/pytorch-sentiment-analysis/.data/imdb/aclImdb/train",
                 glove_file_path="/Users/miaohua/Desktop/myfavor/pytorch-sentiment-analysis/.vector_cache/glove.6B.100d.txt",
                 model_path="/Users/miaohua/Desktop/myfavor/pytorch-sentiment-analysis/gpu/SentimentImdb/model_storage/model_lstm.pth",
                 vect_file_path="/Users/miaohua/Desktop/myfavor/pytorch-sentiment-analysis/gpu/SentimentImdb/model_storage/vect_file.dmp",
                 model='lstm',
                 tokenizer='spacy',
                 bert_model='bert-base-uncased',
                 cuda=False,
                 seed=1337,
                 learning_rate=1e-3,
                 batch_size=64,
                 num_epochs=5,
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

if __name__ == '__main__':
    # args
    parser = args_parser()
    args = parser.parse_args()
    args = set_args(args, running_args)

    if args.train:
        print('exec training routine.......................................')
        trainer = Trainer(args)
        trainer.pre_train()
        trainer.train()
        trainer.after_train()
    
    if args.test:
        print('exec testing routine........................................')
        assert args.load_model==True, 'when testing, load_model must be True'
        tester = Tester(args)
        tester.pre_test()
        tester.test()


