from Transformer.Attention import PositionWiseEncoder
from Transformer.EncodeLayer import EncodeLayer
from torch import nn



class TransEncoder(nn.Module):
    def __init__(self, input_size, embedding_dim, ff_hidden_dim, max_seq_len, heads=8, enc_layer=6, embedding_weights=None, dropout=0.5, padding_idx=0):
        super(TransEncoder, self).__init__()

        if embedding_weights is None:
            self._embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        else:
            self._embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, padding_idx=padding_idx, _weight=embedding_weights)

        self._pos_enc = PositionWiseEncoder(max_seq_len, embedding_dim)

        self._enc_layers = nn.ModuleList([EncodeLayer(embedding_dim, ff_hidden_dim, heads, dropout) for _ in range(enc_layer)])


    def forward(self, x, mask=None):
        embed_x = self._embedding(x)
        inter_x = self._pos_enc(embed_x)

        for enc_layer in self._enc_layers:
            inter_x, attention = enc_layer(inter_x, mask)

        
        return inter_x, attention
