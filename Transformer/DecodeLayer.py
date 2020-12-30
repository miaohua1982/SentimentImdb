from Transformer.Attention import FeedForwardLayer, MultiHeadAttention
from torch import nn
import torch

class DecodeLayer(nn.Module):
    def __init__(self, input_size, hidden_dim, ff_hidden_dim,  heads=8, dropout=0.5):
        super(DecodeLayer, self).__init__()

        self._q_linear = nn.Linear(hidden_dim, hidden_dim)
        self._k_linear = nn.Linear(hidden_dim, hidden_dim)
        self._v_linear = nn.Linear(hidden_dim, hidden_dim)

        self._self_attention = MultiHeadAttention(heads, hidden_dim, dropout)
        self._enc_attention = MultiHeadAttention(heads, hidden_dim, dropout)
        self._att_layer_norm = nn.LayerNorm(hidden_dim)
        self._ffn = FeedForwardLayer(hidden_dim, ff_hidden_dim, dropout)
        self._ffn_layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, enc_output, mask=None):
        # self attention
        q = self._q_linear(x)
        k = self._k_linear(x)
        v = self._v_linear(x)
        qv, attention = self._self_attention(q, k, v, mask)
        atten_x = self._att_layer_norm(qv+x)
        # encoder attention
        enc_qv, attention = self._enc_attention(qv, enc_output, enc_output, mask)
        enc_x = self._att_layer_norm(enc_qv+qv)
        # feed forward
        inter_x = self._ffn(enc_x)

        output = self._ffn_layer_norm(enc_x+inter_x)

        return output, attention
