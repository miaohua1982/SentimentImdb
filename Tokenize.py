import spacy
import re
import numpy as np
from collections import Counter
from transformers import BertTokenizer
from Vocab import SeqVocabulary

class BaseTokenize(object):
    def __init__(self):
        self._tokenizer_name_ = 'base'
        self._pad_token = 'PAD'
        self._unk_token = 'UNK'
        self._mask_token = 'MASK'
        self._cls_token = 'CLS'

        self._words_vocab = SeqVocabulary(pad_token=self._pad_token, unk_token=self._unk_token, mask_token=self._mask_token, cls_token=self._cls_token)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
        return text

    def tokenizer(self, text):
        text = self.preprocess_text(text)
        return text.split(' ')

    def do_vectorize(self, text):
        tokens_inds = [self._words_vocab.lookup_token(one) for one in self.tokenizer(text)]
        return tokens_inds

    def build_vocab(self, sentiment_df, max_vocab_len=-1):
        if self.is_need_build_vocab():
            words_count = Counter()
            def do_words_count(sentiment, words_count):
                #for one in sentiment.split(' '):
                for one in self.tokenizer(sentiment):
                    words_count[one] += 1

            _ = sentiment_df.sentiment.apply(lambda s: do_words_count(s, words_count))

            sorted_wc = sorted(words_count.items(), key=lambda i:i[1], reverse=True)
            sorted_wc = [one[0] for one in sorted_wc]
            if max_vocab_len < 0:
                self._words_vocab.add_tokens(sorted_wc)
            else:
                self._words_vocab.add_tokens(sorted_wc[:max_vocab_len])

    def vectorize(self, sentence, max_vector_len=-1):
        tokens_inds = self.do_vectorize(sentence)
        
        if max_vector_len<0:
            max_vector_len = len(tokens_inds)+2  # start & end token

        source_rep = np.zeros(max_vector_len, dtype=np.int64)
                
        if max_vector_len < len(tokens_inds):
            source_rep[:] = tokens_inds[:max_vector_len]
        else:
            source_rep[:len(tokens_inds)] = tokens_inds
            source_rep[len(tokens_inds):] = self.get_pad_ind()
        
        return source_rep, len(tokens_inds)

    def get_pad_ind(self):
        return self._words_vocab.get_pad_ind()

    def get_mask_ind(self):
        return self._words_vocab.get_mask_ind()

    def get_start_ind(self):
        return self._words_vocab.get_seq_start_ind()

    def get_end_ind(self):
        return self._words_vocab.get_seq_end_ind()

    def get_words_vocab(self):
        return self._words_vocab
    
    def is_need_build_vocab(self):
        return True

class SpacyTokenize(BaseTokenize):
    def __init__(self, lang):
        super(SpacyTokenize, self).__init__()
        self._token_name_ = 'spacy'
        self._lang = lang
        self.nlp = spacy.load(lang)
    
    def tokenizer(self, text):
        return [one.text for one in self.nlp.tokenizer(text)]


class BertTokenize(BaseTokenize):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertTokenize, self).__init__()
        self._token_name_ = 'bert'
        self.nlp = BertTokenizer.from_pretrained(model_name)
        self._unk_token = self.nlp.unk_token
        self._mask_token = self.nlp.mask_token
        self._cls_token = self.nlp.cls_token

    def tokenizer(self, text):
        tokens = self.nlp.tokenize(text)
        return tokens

class AdBertTokenize(BaseTokenize):
    def __init__(self, model_name='bert-base-uncased'):
        super(AdBertTokenize, self).__init__()
        self._token_name_ = 'adbert'
        self.nlp = BertTokenizer.from_pretrained(model_name)
        self._model_name = model_name
        self._unk_token = self.nlp.unk_token
        self._mask_token = self.nlp.mask_token
        self._cls_token = self.nlp.cls_token
        self._pad_token = self.nlp.pad_token

    def tokenizer(self, text):
        tokens = self.nlp.tokenize(text)
        return tokens

    def do_vectorize(self, text):
        tokens = self.tokenizer(text)
        tokens_inds = self.nlp.convert_tokens_to_ids(tokens)
        return tokens_inds

    def get_mask_ind(self):
        return self.nlp.convert_tokens_to_ids(self._mask_token)
    
    def get_pad_ind(self):
        return self.nlp.convert_tokens_to_ids(self._pad_token)

    def get_start_ind(self):
        return self.nlp.convert_tokens_to_ids(self.nlp.bos_token)

    def get_end_ind(self):
        return self.nlp.convert_tokens_to_ids(self.nlp.eos_token)

    def get_max_len(self):
        return self.nlp.max_model_input_sizes[self._model_name]
    
    def is_need_build_vocab(self):
        return False