from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    def __init__(self, tokenizer, train_ds=None, valid_ds=None, test_ds=None):
        if train_ds is not None:
            self._train_df = train_ds
            self._train_df_size = self._train_df.shape[0]
        
        if valid_ds is not None:
            self._val_df = valid_ds
            self._val_df_size = self._val_df.shape[0]
        
        if test_ds is not None:
            self._test_df = test_ds
            self._test_df_size = self._test_df.shape[0]
    
        self._tokenizer = tokenizer
        
        self._max_length = train_ds.sentiment.apply(lambda s: len(tokenizer.tokenizer(s))).max()
                
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
        source_vec, source_len = self._tokenizer.vectorize(self._target_df.iloc[ind].sentiment, self._max_length)
        return source_vec, source_len, self._target_df.iloc[ind].flag
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size
    
    def get_max_len(self):
        return self._max_length

    def set_max_len(self, max_len):
        self._max_length = max_len
         
