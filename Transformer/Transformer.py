from Transformer.TransEncoder import TransEncoder
from Transformer.TransDecoder import TransDecoder
from torch import nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, ff_hidden_dim, max_seq_ln, classes_num, heads=8, enc_layer=6, dec_layer=6, embedding_weights=None, dropout=0.5, padding_idx=0):
        self._enc = TransEncoder(input_size, embedding_size, hidden_size, ff_hidden_dim, max_seq_len, heads, enc_layer, embedding_weights, dropout, padding_idx)
        self._dec = TransEncoder(input_size, embedding_size, hidden_size, ff_hidden_dim, max_seq_len, heads, enc_layer, embedding_weights, dropout, padding_idx)
        
        self._classifer = nn.Linear(hidden_size, classes_num)

    def forward(self, x, enc_mask=None, dec_mask=None):
        enc_output, enc_attention = self._enc(x, enc_mask) 
        dec_output, dec_attention = self._dec(x, enc_output, dec_mask)
        
        output = F.softmax(self._classifier(dec_output), dim=-1)

        return output, enc_attention, dec_attention

                
        
        
