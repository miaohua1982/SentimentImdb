from Transformer.Attention import FeedForwardLayer, MultiHeadAttention
from torch import nn
import torch.nn.functional as F

      
class EncodeLayer(nn.Module):
    def __init__(self, hidden_dim, ff_hidden_dim,  heads=8, dropout=0.5):
        super().__init__()

        self._q_linear = nn.Linear(hidden_dim, hidden_dim)
        self._k_linear = nn.Linear(hidden_dim, hidden_dim)
        self._v_linear = nn.Linear(hidden_dim, hidden_dim)

        self._attention = MultiHeadAttention(heads, hidden_dim, dropout)
        self._att_layer_norm = nn.LayerNorm(hidden_dim)
        self._ffn = FeedForwardLayer(hidden_dim, ff_hidden_dim, dropout)
        self._ffn_layer_norm = nn.LayerNorm(hidden_dim)

        self._dropout = dropout
        
    def forward(self, x, mask=None):
        
        q = self._q_linear(x)
        k = self._k_linear(x)
        v = self._v_linear(x)
        qv, attention = self._attention(q, k, v, mask)
        
        atten_x = self._att_layer_norm(F.dropout(qv, p=self._dropout)+x)

        inter_x = self._ffn(atten_x)

        output = self._ffn_layer_norm(atten_x+inter_x)

        return output, attention

        
