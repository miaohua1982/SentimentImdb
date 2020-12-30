
class Vocabulary(object):
    def __init__(self):
        self._voc_2_ind = {}
        self._ind_2_voc = {}
    
    def add_token(self, token):
        if token in self._voc_2_ind:
            return self._voc_2_ind[token]
        else:
            ind = len(self._voc_2_ind)
            self._voc_2_ind[token] = ind
            self._ind_2_voc[ind] = token
            return ind
    
    def add_tokens(self, tokens):
        return [self.add_token(t) for t in tokens]            
    
    def lookup_token(self, token):
        if token not in self._voc_2_ind:
            return KeyError("the token (%s) is not in the Vocabulary" % token)
        else:
            return self._voc_2_ind[token]

    def lookup_ind(self, ind):
        if ind not in self._ind_2_voc:
            raise KeyError("the index (%d) is not in the Vocabulary" % ind)
        return self._ind_2_voc[ind]
    
    def __len__(self):
        return len(self._voc_2_ind)

class SeqVocabulary(Vocabulary):
    def __init__(self, pad_token='PAD', unk_token='UNK', mask_token='MASK', cls_token='CLS', seq_start_token=None, seq_end_token=None):
        super(SeqVocabulary, self).__init__()
        
        self._pad_token = pad_token
        self._unk_token = unk_token
        self._mask_token = mask_token
        self._cls_token = cls_token
        
        self._pad_ind = self.add_token(pad_token)
        self._unk_ind = self.add_token(unk_token)
        self._mask_ind = self.add_token(mask_token)
        self._cls_ind = self.add_token(cls_token)
        
        self._seq_start_token = seq_start_token
        self._seq_end_token = seq_end_token
        
        if seq_start_token is not None:
            self._seq_start_token_idx = self.add_token(seq_start_token)
        if seq_end_token is not None:
            self._seq_end_token_idx = self.add_token(seq_end_token)
    
    def lookup_token(self, token):
        return self._voc_2_ind.get(token, self._unk_ind)
        
    def get_pad_ind(self):
        return self._pad_ind
    
    def get_unk_ind(self):
        return self._unk_ind
    
    def get_mask_ind(self):
        return self._mask_ind
 
    def get_cls_ind(self):
        return self._cls_ind
    
    def get_seq_start_ind(self):
        if self._seq_start_token is not None:
            return self._seq_start_token_idx
    
    def get_seq_end_ind(self):
        if self._seq_end_token is not None:
            return self._seq_end_token_idx
