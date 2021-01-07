from Transformer import TransEncoder
import torch
from torch import nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_len, embedding_dim, ff_hidden, max_seq_len, classes_num=1, heads=8, enc_layers=6, embedding_weights=None, dropout=0.5, padding_idx=0, bidirectional=True):
        super(TransformerClassifier, self).__init__()
        
        self._enc = TransEncoder(vocab_len, embedding_dim, ff_hidden, max_seq_len, heads, enc_layers, embedding_weights, dropout, padding_idx)
        if bidirectional:
            self._reverse_enc = TransEncoder(vocab_len, embedding_dim, ff_hidden, max_seq_len, heads, enc_layers, embedding_weights, dropout, padding_idx)
        self._bidirectional = bidirectional

        #self._forward = nn.Linear(embedding_dim, ff_hidden)
        self._classifier = nn.Linear(embedding_dim, classes_num)
        
    def forward(self, x_data, *args, **kwargs):
        enc_x, attention = self._enc(x_data)    # batch size, seq length, embedding_dim
        if self._bidirectional:
            r_enc_x, r_attetion = self._reverse_enc(torch.flip(x_data, (-1,))) # flip last dimention
            enc_x = torch.cat([enc_x, r_enc_x], dim=1)  # batch size, seq length*2, embedding_dim
        #inter_x = enc_x[:,0,:]  # batch size, embedding_dim
        inter_x = F.avg_pool2d(enc_x, (enc_x.shape[1], 1)).squeeze(1)  # batch size, embedding_dim
        
        output = self._classifier(inter_x)
        return output
