import numpy as np
from collections import Counter
from Vocab import SeqVocabulary

class SentimentVectorizer(object):
    def __init__(self, nlp_tokenizer):
        self._nlp_tokenizer = nlp_tokenizer

    def vectorize(self, sentence, max_vector_len=-1):
        tokens_inds = self._nlp_tokenizer.vectorize(sentence)
        
        if max_vector_len<0:
            max_vector_len = len(tokens_inds)+2  # start & end token

        source_rep = np.zeros(max_vector_len, dtype=np.int64)
                
        if max_vector_len < len(tokens_inds):
            source_rep[:] = tokens_inds[:max_vector_len]
        else:
            source_rep[:len(tokens_inds)] = tokens_inds
            source_rep[len(tokens_inds):] = self._nlp_tokenizer.get_pad_ind()
        
        return source_rep, len(tokens_inds)

    def get_tokenizer(self):
        return self._nlp_tokenizer
    
    @classmethod
    def from_dataframe(cls, sentiment_df, nlp_tokenizer, max_vocab_len=25000):
        words_vocab = nlp_tokenizer.get_words_vocab()
        words_count = Counter()
        def do_words_count(sentiment, words_count):
            #for one in sentiment.split(' '):
            for one in nlp_tokenizer.tokenizer(sentiment):
                words_count[one] += 1

        _ = sentiment_df.sentiment.apply(lambda s: do_words_count(s, words_count))

        sorted_wc = sorted(words_count.items(), key=lambda i:i[1], reverse=True)
        sorted_wc = [one[0] for one in sorted_wc]
        if max_vocab_len < 0:
            words_vocab.add_tokens(sorted_wc)
        else:
            words_vocab.add_tokens(sorted_wc[:max_vocab_len])

        return cls(nlp_tokenizer)

