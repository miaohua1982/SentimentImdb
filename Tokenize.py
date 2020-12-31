import spacy
import re
from transformers import BertTokenizer
from Vocab import SeqVocabulary

class BaseTokenize(object):
    def __init__(self):
        self._tokenizer_name_ = 'base'
        self._unk_token = 'UNK'
        self._mask_token = 'MASK'
        self._cls_token = 'CLS'

        self._words_vocab = SeqVocabulary(unk_token=self._unk_token, mask_token=self._mask_token, cls_token=self._cls_token)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
        return text

    def tokenizer(self, text):
        text = self.preprocess_text(text)
        return text.split(' ')

    def vectorize(self, text):
        tokens_inds = [self._words_vocab.lookup_token(one) for one in self.tokenizer(text)]
        return tokens_inds

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
        self.nlp = BertTokenizer.from_pretrained(model_name)
        self._model_name = model_name
        self._unk_token = self.nlp.unk_token
        self._mask_token = self.nlp.mask_token
        self._cls_token = self.nlp.cls_token
        self._pad_token = self.nlp.pad_token

    def tokenizer(self, text):
        tokens = self.nlp.tokenize(text)
        return tokens

    def vectorize(self, text):
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