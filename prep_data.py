from torch_datasets.dataset import MovieRelationDataset
from torch_datasets.vectorizer import IOVectorizer, OneHotVocabVectorizer, VocabLookupVectorizer, \
 SequenceVocabVectorizer, OneHotSequenceVocabVectorizer, BertVectorizer
from torch_datasets.vocabulary import Vocabulary

import numpy as np
import torch
import pickle as pkl
import pandas as pd

class SingleLabelDatasets(object):
    def __init__(self, dataset_dict, input_vocabulary, output_vocabulary):
        
        input_vectorizer = OneHotVocabVectorizer(input_vocabulary) # input vectorizer, one-hot or sequence
        output_vectorizer = VocabLookupVectorizer(output_vocabulary) # output vectorizer, index lookup

        self.io_vectorizer = io_vectorizer = IOVectorizer(input_vectorizer, output_vectorizer)

        self.train_dataset = MovieRelationDataset(dataset_dict["train"], io_vectorizer)
        self.validation_dataset = MovieRelationDataset(dataset_dict["val"], io_vectorizer)
        self.test_dataset = MovieRelationDataset(dataset_dict["test"], io_vectorizer)
        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary


    @classmethod
    def from_train_dataframe(cls, train_df):

        train_data = train_df[train_df.split == "train"]
        val_data = train_df[train_df.split == "val"]
        test_data = train_df[train_df.split == "test"]

        input_vocabulary = Vocabulary.from_documents(train_data.UTTERANCE.tolist()) # input vocabulary, must comprise of utterances; can replace with word2vec vocabulary later
        output_vocabulary = Vocabulary.from_documents(train_df.primary_relation.tolist(), add_unk=False) # output vocabulary, must comprise of our labels (all possible labels in our dataset)
        data_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data
            }


        return cls(data_dict, input_vocabulary, output_vocabulary)

    def get_input_vocabulary(self):
        return self.input_vocabulary

    def get_output_vocabulary(self):
        return self.output_vocabulary

    def save_state_obj(self, path):
        torch.save(self.io_vectorizer, path)

class OneHotMultilabelDatasets(object):

    def __init__(self, dataset_dict, input_vocabulary, output_vocabulary):
        
        input_vectorizer = OneHotVocabVectorizer(input_vocabulary) # input vectorizer onehot
        output_vectorizer = OneHotVocabVectorizer(output_vocabulary) # output vectorizer onehot
        self.dataset_dict = dataset_dict

        self.io_vectorizer = io_vectorizer = IOVectorizer(input_vectorizer, output_vectorizer)

        self.train_dataset = MovieRelationDataset(dataset_dict["train"], io_vectorizer)
        self.validation_dataset = MovieRelationDataset(dataset_dict["val"], io_vectorizer)
        self.test_dataset = MovieRelationDataset(dataset_dict["test"], io_vectorizer)
        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary
        self.set_vectorizer(io_vectorizer)

    def set_vectorizer(self, io_vectorizer):
        self.train_dataset = MovieRelationDataset(self.dataset_dict["train"], self.io_vectorizer)
        self.validation_dataset = MovieRelationDataset(self.dataset_dict["val"], self.io_vectorizer)
        self.test_dataset = MovieRelationDataset(self.dataset_dict["test"], self.io_vectorizer)

    @classmethod
    def from_train_dataframe(cls, train_df):

        train_data = train_df[train_df.split == "train"]
        val_data = train_df[train_df.split == "val"]
        test_data = train_df[train_df.split == "test"]

        input_vocabulary = Vocabulary.from_documents(train_data.UTTERANCE.tolist()) # input vocabulary, must comprise of utterances; can replace with word2vec vocabulary later
        output_vocabulary = Vocabulary.from_documents(train_df.RELATIONS.tolist(), add_unk=False) # output vocabulary, must comprise of our labels (all possible labels in our dataset)
        data_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data
            }


        return cls(data_dict, input_vocabulary, output_vocabulary)

    def get_input_vocabulary(self):
        return self.input_vocabulary

    def get_output_vocabulary(self):
        return self.output_vocabulary

    def save_state_obj(self, path):
        torch.save(self.io_vectorizer, path)


class EmbeddedSequenceMultilabelDatasets(object):
    def __init__(self, dataset_dict, input_vocabulary, output_vocabulary, seq_len):
        input_vectorizer = SequenceVocabVectorizer(input_vocabulary, seq_len) # input vectorizer sequence
        output_vectorizer = OneHotVocabVectorizer(output_vocabulary) # output vectorizer onehot
        self.dataset_dict = dataset_dict
        self.io_vectorizer = io_vectorizer = IOVectorizer(input_vectorizer, output_vectorizer)

        self.set_vectorizer(io_vectorizer)
        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary

    def set_vectorizer(self, io_vectorizer):
        self.io_vectorizer = io_vectorizer
        self.train_dataset = MovieRelationDataset(self.dataset_dict["train"], self.io_vectorizer)
        self.validation_dataset = MovieRelationDataset(self.dataset_dict["val"], self.io_vectorizer)
        self.test_dataset = MovieRelationDataset(self.dataset_dict["test"], self.io_vectorizer)

    @classmethod
    def from_train_dataframe(cls, train_df, dim):

        train_data, val_data = train_test_split(train_df[train_df.split != "test"], shuffle=True, test_size=0.177)
        test_data = train_df[train_df.split == "test"]


        with open('processing/dictionary_{}d.pkl'.format(dim), 'rb') as dictionary:
            word2vec_dict = pkl.load(dictionary)

        input_vocabulary = Vocabulary.from_dict(word2vec_dict['word2id'], add_unk=True, unk_token="unk") # input vocabulary, must comprise of utterances; can replace with word2vec vocabulary later
        output_vocabulary = Vocabulary.from_documents(train_df.RELATIONS.tolist(), add_unk=False) # output vocabulary, must comprise of our labels (all possible labels in our dataset)
        data_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data
            }

        max_seq_len = max([len(row.split(" ")) for row in train_df.UTTERANCE.tolist()])

        return cls(data_dict, input_vocabulary, output_vocabulary, max_seq_len)

    def get_input_vocabulary(self):
        return self.input_vocabulary

    def get_output_vocabulary(self):
        return self.output_vocabulary

    def save_state_obj(self, path):
        torch.save(self.io_vectorizer, path)

class OneHotSequenceMultilabelDatasets(object):
    def __init__(self, dataset_dict, input_vocabulary, output_vocabulary, seq_len):
        
        input_vectorizer = OneHotSequenceVocabVectorizer(input_vocabulary, seq_len) # input vectorizer sequence
        output_vectorizer = OneHotVocabVectorizer(output_vocabulary) # output vectorizer onehot
        self.dataset_dict = dataset_dict
        self.io_vectorizer = io_vectorizer = IOVectorizer(input_vectorizer, output_vectorizer)
        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary
        self.set_vectorizer(io_vectorizer)

    def set_vectorizer(self, io_vectorizer):
        self.io_vectorizer = io_vectorizer
        self.train_dataset = MovieRelationDataset(self.dataset_dict["train"], self.io_vectorizer)
        self.validation_dataset = MovieRelationDataset(self.dataset_dict["val"], self.io_vectorizer)
        self.test_dataset = MovieRelationDataset(self.dataset_dict["test"], self.io_vectorizer)


    @classmethod
    def from_train_dataframe(cls, train_df):
        train_df.UTTERANCE =  "<S> " + train_df.UTTERANCE + " </S>"

        train_data, holdout_data = train_test_split(train_df, shuffle=True, test_size=0.3)

        val_data, test_data = train_test_split(holdout_data, test_size=0.5)
        
        input_vocabulary = Vocabulary.from_documents(train_data.UTTERANCE.tolist()) # input vocabulary, must comprise of utterances; can replace with word2vec vocabulary later
        output_vocabulary = Vocabulary.from_documents(train_df.RELATIONS.tolist(), add_unk=False) # output vocabulary, must comprise of our labels (all possible labels in our dataset)
        data_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data
            }
        
        max_seq_len = max([len(row.split(" ")) for row in train_data.UTTERANCE.tolist()])

        return cls(data_dict, input_vocabulary, output_vocabulary, max_seq_len)


    def get_input_vocabulary(self):
        return self.input_vocabulary

    def get_output_vocabulary(self):
        return self.output_vocabulary

    def save_state_obj(self, path):
        torch.save(self.io_vectorizer, path)

class BertTokenizedMultilabelDatasets(object):
    def __init__(self, dataset_dict, output_vocabulary, seq_len):
        input_vectorizer = BertVectorizer()

        output_vectorizer = OneHotVocabVectorizer(output_vocabulary) # output vectorizer onehot

        self.io_vectorizer = io_vectorizer = IOVectorizer(input_vectorizer, output_vectorizer)

        self.train_dataset = MovieRelationDataset(dataset_dict["train"], io_vectorizer)
        self.validation_dataset = MovieRelationDataset(dataset_dict["val"], io_vectorizer)
        self.test_dataset = MovieRelationDataset(dataset_dict["test"], io_vectorizer)
        self.output_vocabulary = output_vocabulary


    @classmethod
    def from_train_dataframe(cls, train_df):
        train_df.UTTERANCE =  "<S> " + train_df.UTTERANCE + " </S>"

        train_data, holdout_data = train_test_split(train_df, shuffle=True, test_size=0.2)

        val_data, test_data = train_test_split(holdout_data, test_size=0.1)
        
        input_vocabulary = Vocabulary.from_documents(train_data.UTTERANCE.tolist()) # input vocabulary, must comprise of utterances; can replace with word2vec vocabulary later
        output_vocabulary = Vocabulary.from_documents(train_df.RELATIONS.tolist(), add_unk=False) # output vocabulary, must comprise of our labels (all possible labels in our dataset)
        data_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data
            }
        
        max_seq_len = max([len(row.split(" ")) for row in train_data.UTTERANCE.tolist()])

        return cls(data_dict, output_vocabulary, max_seq_len)

    def get_output_vocabulary(self):
        return self.output_vocabulary

    def save_state_obj(self, path):
        torch.save(self.io_vectorizer, path)


from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    train_df = pd.read_csv('data/hw1_train.csv')
    relations = [relation.split(" ")[0] for relation in train_df.RELATIONS.tolist()]
    train_df['primary_relation'] = relations

    train_data, holdout_data = train_test_split(train_df, shuffle=True, test_size=0.3)

    validation_data, test_data = train_test_split(holdout_data, test_size=0.5)
    
    train_data['split'] = 'train'
    validation_data['split'] = 'val'
    test_data['split'] = 'test'

    prepared_df = pd.concat([train_data, validation_data, test_data])
    print(prepared_df.groupby(['primary_relation']).count())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        print(prepared_df.groupby(['split', 'primary_relation']).count())
    prepared_df.to_csv('data/train.csv')