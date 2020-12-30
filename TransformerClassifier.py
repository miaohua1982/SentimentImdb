from Transformer import TransEncoder
from torch import nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_len, embedding_dim, ff_hidden, max_seq_len, classes_num=1, heads=8, enc_layers=6, embedding_weights=None, dropout=0.5, padding_idx=0):
        super(TransformerClassifier, self).__init__()
        
        self._enc = TransEncoder(vocab_len, embedding_dim, ff_hidden, max_seq_len, heads, enc_layers, embedding_weights, dropout, padding_idx)
        #self._forward = nn.Linear(embedding_dim, ff_hidden)
        self._classifier = nn.Linear(embedding_dim, classes_num)
        
    def forward(self, x_data, *args, **kwargs):
        enc_x, attention = self._enc(x_data)    # batch size, seq length, embedding_dim
        #inter_x = enc_x[:,0,:]  # batch size, embedding_dim
        inter_x = F.avg_pool2d(enc_x, (enc_x.shape[1], 1)).squeeze(1)  # batch size, embedding_dim
        
        output = self._classifier(inter_x)
        return output
