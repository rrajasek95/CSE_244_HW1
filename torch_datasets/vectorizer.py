import numpy as np
import string
from transformers import BertTokenizer

class IOVectorizer(object):
    # A vectorizer that does input and output vectorization
    def __init__(self, input_vectorizer, output_vectorizer):
        self.input_vectorizer = input_vectorizer
        self.output_vectorizer = output_vectorizer
    
    def vectorize_input(self, x):
        return self.input_vectorizer.vectorize(x)

    def vectorize_output(self, y):
        return self.output_vectorizer.vectorize(y)

class OneHotVocabVectorizer(object):
    # Vectorizer that accepts a vocabulary of words and performs one-hot encoding of an utterance

    def __init__(self, vocabulary, dtype=np.float32):
        self.vocabulary = vocabulary
        self.dtype = dtype

    def vectorize(self, utterance):
        one_hot = np.zeros(len(self.vocabulary), dtype=self.dtype)

        for token in utterance.split(" "):
            if token not in string.punctuation:
                one_hot[self.vocabulary.lookup_token(token)] = 1
        return one_hot


class SequenceVocabVectorizer(object):

    def __init__(self, vocabulary, sequence_length):
        self.vocabulary = vocabulary
        self.seq_length = sequence_length

    def vectorize(self, utterance):
        sequence = np.zeros(self.seq_length, dtype=np.long)
        idx = 0
        for token in utterance.split(" "):
            if token not in string.punctuation:
                sequence[idx] = self.vocabulary.lookup_token(token)
                idx += 1

        return sequence

class OneHotSequenceVocabVectorizer(object):
    def __init__(self, vocabulary, sequence_length):
        self.vocabulary = vocabulary
        self.seq_length = sequence_length

    def vectorize(self, utterance):
        one_hot_matrix_size = len(self.vocabulary), self.seq_length
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)

        idx = 0
        for token in utterance.split(" "):
            if token not in string.punctuation:
                word_idx = self.vocabulary.lookup_token(token)
                one_hot_matrix[word_idx][idx] = 1
                idx += 1

        return one_hot_matrix

class VocabLookupVectorizer(object):
    def __init__(self, vocabulary, dtype=np.float32):
        self.vocabulary = vocabulary
        self.dtype = dtype

    def vectorize(self, utterance):
        return self.vocabulary.lookup_token(utterance)

class BertVectorizer(object):
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def vectorize(self, utterance):
        return self._tokenizer.encode(utterance, add_special_tokens=True).unsqueeze(0)