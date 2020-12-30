from torch import nn
import torch.nn.functional as F
import math
import torch

class PositionWiseEncoder(nn.Module):
    def __init__(self, max_seq_len, hidden_dim):
       super(PositionWiseEncoder, self).__init__()

       self._max_seq_len = max_seq_len
       self._pos_enc_tb = torch.zeros(max_seq_len, hidden_dim).to('cuda:0' if torch.cuda.is_available() else 'cpu')
       for pos in range(max_seq_len):
           for i in range(0, hidden_dim, 2):
               self._pos_enc_tb[pos,i] = math.sin(pos/10000**(2*i/hidden_dim))
               self._pos_enc_tb[pos,i+1] = math.cos(pos/1000**(2*(i+1)/hidden_dim))

    def forward(self, x):
        L = x.shape[1]
        assert L <= self._max_seq_len, 'Error:Seq length must be less than max seq length'

        return x+self._pos_enc_tb[:L,:]

class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(FeedForwardLayer, self).__init__()

        self._linear = nn.Linear(input_dim, hidden_dim)
        self._forward = nn.Linear(hidden_dim, input_dim)
        self._dropout = dropout

    def forward(self, x):
        inter = F.dropout(F.relu(self._linear(x)), p=self._dropout)
        return self._forward(inter)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_dim, dropout=0.5):
        super(MultiHeadAttention, self).__init__()

        assert hidden_dim%heads == 0

        self._heads = heads
        self._dim_per_head = hidden_dim//heads
        self._hidden_dim = hidden_dim
        self._k = self._dim_per_head**-0.5
        self._dropout = dropout

        self.out = nn.Linear(hidden_dim, hidden_dim)

    
    def forward(self, q, k, v, mask=None):
        qbs, qseq, qdim = q.shape
        kbs, kseq, kdim = k.shape
        vbs, vseq, vdim = v.shape
        
        assert self._hidden_dim == qdim and qdim == kdim and kdim == vdim, 'Error: Q, K, V hidden_dim must be equal to initail parameter'
        assert kseq == vseq, 'Error: sequence length for K must be equal to sequence length for V'

        # batch size, seq length, heads, head_dim
        q = q.view(qbs, qseq, self._heads, self._dim_per_head)
        k = k.view(kbs, kseq, self._heads, self._dim_per_head)
        v = v.view(vbs, vseq, self._heads, self._dim_per_head)
        # batch size, heads, seq_length, head_dim        
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,3,1)
        v = v.permute(0,2,1,3)

        attention = q.matmul(k)  #batch size, heads, q_seq_length, k_seq_length
        attention = attention*self._k
     
        if mask is not None:
             attention.mask_fill_(mask==0, 1e-9)
        attention = F.softmax(attention, dim=-1)
        qv = attention.matmul(v)     # batch size, heads, q_seq_length, dim_per_head
        qv = qv.transpose(2,1).contiguous().view(qbs, qseq, qdim)
        
        return self.out(qv), attention
        

