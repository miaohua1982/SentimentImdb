import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnClassifier(nn.Module):
    def __init__(self, vocab_len, embedding_dim, classes_num, out_channels, filter_size, embedding_weights=None, padding_idx=0, dropout=0.5):
        super(CnnClassifier, self).__init__()

        if embedding_weights is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim, padding_idx=padding_idx, _weight=embedding_weights)
        
        self.cnns = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(filter_size[i],embedding_dim)) for i in range(len(filter_size))])
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_channels*len(filter_size), classes_num)

    def set_embedding_weights(self, ws):
        _ = self.embedding.weight.data.copy_(ws)        

    def forward(self, x_data, *args, **kwargs):
        embed_data = self.embedding(x_data)
        embed_data = embed_data.unsqueeze(dim=1)

        inter_vec = []
        for one_cnn in self.cnns:
            cnn_data = F.relu(one_cnn(embed_data).squeeze(dim=3))    # batch, out_channels, seq_len-filter_size[i]
            cnn_data = F.max_pool1d(cnn_data, kernel_size=cnn_data.shape[2]).squeeze(dim=2)  # batch, out_channels
            inter_vec.append(cnn_data)

        inter_vec = torch.cat(inter_vec,dim=1)    # batch, out_channels*len(filter_size)

        return self.classifier(self.dropout(inter_vec))
