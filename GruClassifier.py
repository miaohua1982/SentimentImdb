import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GruClassifier(nn.Module):
    def __init__(self, vocab_len, embedding_dim, rnn_hidden_dim, classes_num, \
                 embedding_weights = None, rnn_layers=3, bidirectional=True, padding_idx=0, dropout=0.5):
        super(GruClassifier, self).__init__()

        if embedding_weights is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim, padding_idx=padding_idx, _weight=embedding_weights)
        
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_layers, \
                          bidirectional=bidirectional, dropout=dropout)

        if bidirectional:
            rnn_hidden_dim = rnn_hidden_dim*2
        self.classifier = nn.Linear(rnn_hidden_dim, classes_num)

        self._dropout = dropout
        self._bidirectional = bidirectional    

    def set_embedding_weights(self, ws):
        _ = self.embedding.weight.data.copy_(ws)

    def forward(self, x_data, x_len):
        embed_data = F.dropout(self.embedding(x_data), p=self._dropout)

        packed_data = pack_padded_sequence(embed_data, x_len, batch_first=True)

        output, hn = self.gru(packed_data)

        if self._bidirectional: 
            inter_vec = torch.cat([hn[-2,:,:], hn[-1,:,:]], dim=1)
        else:
            inter_vec = hn[-1, :, :]

        pred = self.classifier(F.dropout(inter_vec, p=self._dropout))

        return pred
    
