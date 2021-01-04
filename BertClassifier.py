from transformers import BertModel
import torch
from torch import nn
import torch.nn.functional as F

class BertClassifier(nn.Module):
    def __init__(self, vocab_len, hidden_dim, num_layers, embedding_dim, embedding_weights=None, classes_num=1, dropout=0.5, padding_idx=0):
        super(BertClassifier, self).__init__()
        
        if embedding_weights is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embedding_dim, padding_idx=padding_idx, _weight=embedding_weights)
 
        self._gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self._classifier = nn.Linear(hidden_dim*2, classes_num)
        
    def forward(self, x_data, *args, **kwargs):
        embed_x = self.embedding(x_data)
        # gru_x : batch size, seq length, hidden_dim*num_directions, hidden: num_layers*num_directions, batch, hidden_dim
        gru_x, hidden = self._gru(embed_x)

        inter_x = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)
        #inter_x = F.avg_pool2d(gru_x, (gru_x.shape[1], 1)).squeeze(1)  # batch size, hidden_dim*num_directions
        
        output = self._classifier(inter_x)
        return output


class AdBertClassifier(nn.Module):
    def __init__(self, hidden_dim, num_layers, bert_model='bert-base-uncased', classes_num=1, dropout=0.5, weight_path=None, config=None):
        super(AdBertClassifier, self).__init__()
        
        self._bert = BertModel.from_pretrained(bert_model)
        #self._bert = BertModel.from_pretrained('../../bert_pretrained/pytorch_model.bin', config='../../bert_pretrained/config.json')
        embedding_dim = self._bert.config.to_dict()['hidden_size']
        self._gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self._classifier = nn.Linear(hidden_dim*2, classes_num)
        
    def forward(self, x_data, *args, **kwargs):
        with torch.no_grad():
            bert_x = self._bert(x_data)[0]
        # gru_x : batch size, seq length, hidden_dim*num_directions, hidden: num_layers*num_directions, batch, hidden_dim
        gru_x, hidden = self._gru(bert_x)

        inter_x = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)
        #inter_x = F.avg_pool2d(gru_x, (gru_x.shape[1], 1)).squeeze(1)  # batch size, hidden_dim*num_directions
        
        output = self._classifier(inter_x)
        return output