from Transformer.Attention import PositionWiseEncoder
from Transformer.DecodeLayer import DecodeLayer
from torch import nn



class TransDecoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_dim, ff_hidden_dim, max_seq_len, device='cpu', heads=8, dec_layer=6, embedding_weights=None, dropout=0.5, padding_idx=0):
        super(TransDecoder, self).__init__()

        if embedding_weights is None:
            self._embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size, padding_idx=padding_idx)
        else:
            self._embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size, padding_idx=padding_idx, _weight=embedding_weights)

        self._input_linear = nn.Linear(embedding_size, hidden_dim)        
        self._pos_enc = PositionWiseEncoder(max_seq_len, hidden_dim)

        self._dec_layers = [DecodeLayer(hidden_dim, ff_hidden_dim, heads, dropout) for _ in range(dec_layer)]


    def forward(self, x, enc_output,  mask=None):
        embed_x = self._embedding(x)

        input_x = self._input_linear(embed_x)
        inter_x = self._pos_enc(input_x)

        for dec_layer in self._dec_layers:
            inter_x, attention = dec_layer(inter_x, enc_output, mask)

        # inter_x shape: batch size, seq length, hidden_dim
        # attention shape: batch size, seq length, seq_length

        return inter_x, attention
