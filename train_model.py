from torch_models.multiclass.perceptron import Perceptron
from torch_models.multiclass.mlp import MultilayerPerceptron
from torch_models.multilabel.mlp_multilabel import MultiLabelMLP
from torch_models.multilabel.deep_fc import DeepFC
from torch_models.multilabel.cnn import OneHotCNN
from torch_models.multilabel.lstm import BiRNN

from train_loop import train_single_label_model, train_multi_label_model
from config import get_perceptron_args, get_mlp_args, \
    get_mlp_multilabel_args, get_deep_fc_args, get_onehot_cnn_args, get_birnn_args, \
    get_bert_args

import torch
import torch.optim as optim

import pandas as pd
import os
from prep_data import SingleLabelDatasets, OneHotMultilabelDatasets, OneHotSequenceMultilabelDatasets, EmbeddedSequenceMultilabelDatasets
import gensim

def load_train_df(args):
    return pd.read_csv(args.train_csv)

def load_glove_weights(dim, device="cpu"):
    model = gensim.models.KeyedVectors.load_word2vec_format('word2vec/glove.6B.{}d.word2vec.txt'.format(dim))
    return torch.FloatTensor(model.vectors)

def train_mlp():
    args = get_mlp_args()
    train_df = load_train_df(args)
    datasets = SingleLabelDatasets.from_train_dataframe(train_df)

    model = MultilayerPerceptron(
        input_dim=len(datasets.get_input_vocabulary()),
        hidden_dim=args.hidden_dim,
        output_dim=len(datasets.get_output_vocabulary()))
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_single_label_model(model, optimizer, datasets, args)
    datasets.save_state_obj(os.path.join(args.save_dir, "vectorizers.pth"))

def train_perceptron():
    args = get_perceptron_args()
    train_df = load_train_df(args)
    datasets = SingleLabelDatasets.from_train_dataframe(train_df)

    model = Perceptron(len(datasets.get_input_vocabulary()), len(datasets.get_output_vocabulary()))
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_single_label_model(model, optimizer, datasets, args)
    datasets.save_state_obj(os.path.join(args.save_dir, "vectorizers.pth"))

def train_multilabel_mlp():
    args = get_mlp_multilabel_args()
    train_df = load_train_df(args)
    datasets = OneHotMultilabelDatasets.from_train_dataframe(train_df)

    model = MultiLabelMLP(
        input_dim=len(datasets.get_input_vocabulary()),
        hidden_dim=args.hidden_dim,
        out_dim=len(datasets.get_output_vocabulary()))
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_multi_label_model(model, optimizer, datasets, args)
    datasets.save_state_obj(os.path.join(args.save_dir, "vectorizers.pth"))

def train_deep_fc():
    args = get_deep_fc_args()
    train_df = load_train_df(args)
    datasets = OneHotMultilabelDatasets.from_train_dataframe(train_df)

    model = DeepFC(
        input_dim=len(datasets.get_input_vocabulary()),
        h1_dim=args.h1_dim,
        h2_dim=args.h2_dim,
        output_dim=len(datasets.get_output_vocabulary()))
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_multi_label_model(model, optimizer, datasets, args)
    datasets.save_state_obj(os.path.join(args.save_dir, "vectorizers.pth"))

def train_onehot_cnn():
    args = get_onehot_cnn_args()
    train_df = load_train_df(args)
    datasets = OneHotSequenceMultilabelDatasets.from_train_dataframe(train_df)

    model = OneHotCNN(
        initial_num_channels=len(datasets.get_input_vocabulary()),
        num_channels=args.num_channels,
        output_dim=len(datasets.get_output_vocabulary()))

    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_multi_label_model(model, optimizer, datasets, args)
    datasets.save_state_obj(os.path.join(args.save_dir, "vectorizers.pth"))

def train_embedding_cnn():
    pass

def train_birnn():
    args = get_birnn_args()
    train_df = load_train_df(args)
    datasets = EmbeddedSequenceMultilabelDatasets.from_train_dataframe(train_df, args.embed_dim)
    print("Loading GloVe Weights")
    weights = load_glove_weights(args.embed_dim, args.device)
    print("GloVe Weights loaded")

    model = BiRNN(
        vocab_size=len(datasets.get_input_vocabulary()),
        embed_dim=args.embed_dim,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        output_dim=len(datasets.get_output_vocabulary()),
        embeddings=weights)

    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_multi_label_model(model, optimizer, datasets, args)
    datasets.save_state_obj(os.path.join(args.save_dir, "vectorizers.pth"))

def train_kfold_birnn():
    args = get_birnn_args()
    print("Loading GloVe Weights")
    weights = load_glove_weights(args.embed_dim)
    print("GloVe Weights loaded")
    for i in range(5):
        args = get_birnn_args()
        args.model_name += str(i)
        args.model_state_file += str(i)

        train_df = load_train_df(args)
        datasets = EmbeddedSequenceMultilabelDatasets.from_train_dataframe(train_df, dim=args.embed_dim)
        

        model = BiRNN(
            vocab_size=len(datasets.get_input_vocabulary()),
            embed_dim=args.embed_dim,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            output_dim=len(datasets.get_output_vocabulary()),
            embeddings=weights)

        model.to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        train_multi_label_model(model, optimizer, datasets, args)
        datasets.save_state_obj(os.path.join(args.save_dir, "vectorizers_{}.pth".format(i)))

def train_bert_uncased():
    args = get_bert_args()

    train_df = load_train_df(args)
    datasets = EmbeddedSequenceMultilabelDatasets.from_train_dataframe(train_df)
        

    model = BiRNN(
        output_dim=len(datasets.get_output_vocabulary()))
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_multi_label_model(model, optimizer, datasets, args)
    datasets.save_state_obj(os.path.join(args.save_dir, "vectorizers_{}.pth".format(i)))

if __name__ == '__main__':
    train_multilabel_mlp()