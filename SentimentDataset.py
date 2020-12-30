from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    def __init__(self, sentiment_df, vectorizer):
        self._train_df = sentiment_df[sentiment_df.split=='train']
        self._train_df_size = self._train_df.shape[0]
        
        self._val_df = sentiment_df[sentiment_df.split=='val']
        self._val_df_size = self._val_df.shape[0]
        
        self._test_df = sentiment_df[sentiment_df.split=='test']
        self._test_df_size = self._test_df.shape[0]
    
        self._vectorizer = vectorizer
        
        self._max_length = sentiment_df.sentiment.apply(lambda s: len(vectorizer.get_tokenizer().tokenizer(s))).max()
        
        self.set_split('train')
        
    def set_split(self,split='train'):
        if split == 'train':
            self._target_df = self._train_df
            self._target_size = self._train_df_size
        elif split == 'test':
            self._target_df = self._test_df
            self._target_size = self._test_df_size
        else:
            self._target_df = self._val_df
            self._target_size = self._val_df_size

    def __len__(self):
        return self._target_size
    
    def __getitem__(self, ind):
        source_vec, source_len = self._vectorizer.vectorize(self._target_df.iloc[ind].sentiment, self._max_length)
        return source_vec, source_len, self._target_df.iloc[ind].flag
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size
    
    def get_max_len(self):
        return self._max_length

    def set_max_len(self, max_len):
        self._max_length = max_len
         
