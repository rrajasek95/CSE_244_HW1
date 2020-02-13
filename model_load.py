import torch
import os

from torch_models.multilabel.mlp_multilabel import MultiLabelMLP
from torch_models.multilabel.deep_fc import DeepFC
from torch_models.multilabel.cnn import OneHotCNN
from torch_models.multilabel.lstm import BiRNN

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))

def load_vectorizer(vectorizer_path):
    return torch.load(vectorizer_path)


def load_best_model(model, save_path, model_name):
    load_model(model, os.path.join(save_path, "{}_best.pth".format(model_name)))


def get_vectorizer_path(args):
    return os.path.join(args.save_dir, "vectorizers.pth".format(args.model_name))

def load_perceptron(args):
    vectorizer = load_vectorizer(get_vectorizer_path(args))
    model = Perceptron(input_dim=len(vectorizer.input_vectorizer.vocabulary)),
    load_best_model(model, args.save_dir, model_name)

    return vectorizer, model

def load_mlp(args):
    vectorizer = load_vectorizer(get_vectorizer_path(args))
    model = MultiLabelMLP(
        input_dim=len(vectorizer.input_vectorizer.vocabulary),
        hidden_dim=args.hidden_dim,
        out_dim=len(vectorizer.output_vectorizer.vocabulary))
    load_best_model(model, args.save_dir, args.model_name)

    return vectorizer, model

def load_deepfc(args):
    vectorizer = load_vectorizer(get_vectorizer_path(args))
    model = DeepFC(
        input_dim=len(vectorizer.input_vectorizer.vocabulary),
        h1_dim=args.h1_dim,
        h2_dim=args.h2_dim,
        output_dim=len(vectorizer.output_vectorizer.vocabulary)) 
    load_best_model(model, args.save_dir, args.model_name)
    
    return vectorizer, model

def load_onehot_cnn(args):
    vectorizer = load_vectorizer(get_vectorizer_path(args))
    model = OneHotCNN(
        initial_num_channels=len(vectorizer.input_vectorizer.vocabulary),
        num_channels=args.num_channels,
        output_dim=len(vectorizer.output_vectorizer.vocabulary)) 
    load_best_model(model, args.save_dir, args.model_name)

    return vectorizer, model

def load_birnn(args):
    vectorizer = load_vectorizer(get_vectorizer_path(args))

    model = BiRNN(
        vocab_size=len(vectorizer.input_vectorizer.vocabulary),
        embed_dim=args.embed_dim,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        output_dim=len(vectorizer.output_vectorizer.vocabulary))

    load_best_model(model, args.save_dir, args.model_name)
    
    return vectorizer, model