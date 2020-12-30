import torch.nn as nn
import torch.nn.functional as F


class FastTextClassifier(nn.Module):
    def __init__(self, vocab_len, embedding_dim, classes_num, embedding_weights=None, padding_idx=0):
        super(FastTextClassifier, self).__init__()

        if embedding_weights is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim, padding_idx=padding_idx, _weight=embedding_weights)
        
        self.classifier = nn.Linear(embedding_dim, classes_num)

    def set_embedding_weights(self, ws):
        _ = self.embedding.weight.data.copy_(ws)

    def forward(self, x_data, *args, **kwags):
        embed_data = self.embedding(x_data)

        inter_vec = F.avg_pool2d(embed_data, (embed_data.shape[1], 1)).squeeze(1)
        
        pred = self.classifier(inter_vec)

        return pred
    
