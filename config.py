from argparse import Namespace
import torch

def get_lstm_args():
    pass


def get_args():
    args = Namespace(
        # Data and path information
        frequency_cutoff=25,
        model_state_file='perceptron.pth',
        train_csv='data/train.csv',
        save_dir='model_storage/',
        vectorizer_file='vectorizer.json',
        weight_decay=0.0,
        batch_size=16,
        early_stopping_criteria=5,
        learning_rate=0.1,
        num_epochs=100,
        seed=1337,
        # Runtime options omitted for space
    )
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def get_perceptron_args():
    args = Namespace(
        # Data and path information
        frequency_cutoff=25,
        model_name="perceptron",
        model_state_file='perceptron',
        train_csv='data/train.csv',
        save_dir='saved_models/multiclass/perceptron/',
        vectorizer_file='vectorizer.json',
        weight_decay=0.0,
        train_batch_size=16,
        validation_batch_size=256,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=100,
        seed=1337,
        # Runtime options omitted for space
    )
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def get_mlp_args():
    args = Namespace(
        # Data and path information
        frequency_cutoff=25,
        model_name="mlp",
        model_state_file='mlp',
        train_csv='data/train.csv',
        save_dir='saved_models/multiclass/mlp/',
        vectorizer_file='vectorizer.json',
        weight_decay=0.0,
        train_batch_size=16,
        validation_batch_size=256,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=100,
        seed=1337,
        # Runtime options omitted for space
        hidden_dim=100,
        save_every_epoch_n=5
    )
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def get_mlp_multilabel_args():
    args = Namespace(
        # Data and path information
        frequency_cutoff=25,
        model_name="mlp_multilabel",
        model_state_file='mlp_multilabel',
        train_csv='data/train.csv',
        save_dir='saved_models/multilabel/mlp/',
        vectorizer_file='vectorizer.json',
        weight_decay=0.0,
        train_batch_size=16,
        validation_batch_size=256,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=100,
        seed=1337,
        # Runtime options omitted for space
        hidden_dim=100,
        save_every_epoch_n=5
    )
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def get_deep_fc_args():
    args = Namespace(
        # Data and path information
        frequency_cutoff=25,
        model_name="deep_fc",
        model_state_file='deep_fc',
        train_csv='data/train.csv',
        save_dir='saved_models/multilabel/deep_fc/',
        vectorizer_file='vectorizer.json',
        weight_decay=0.0,
        train_batch_size=16,
        validation_batch_size=256,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=100,
        seed=1337,
        # Runtime options omitted for space
        h1_dim=128,
        h2_dim=64,
        save_every_epoch_n=5
    )
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def get_onehot_cnn_args():
    args = Namespace(
        # Data and path information
        frequency_cutoff=25,
        model_name="onehot_cnn",
        model_state_file='onehot_cnn',
        train_csv='data/train.csv',
        save_dir='saved_models/multilabel/onehot_cnn/',
        vectorizer_file='vectorizer.json',
        weight_decay=0.,
        train_batch_size=16,
        validation_batch_size=256,
        early_stopping_criteria=5,
        learning_rate=1e-3,
        num_epochs=100,
        seed=1337,
        # Runtime options omitted for space
        num_channels=256,
        save_every_epoch_n=5
    )
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def get_birnn_args():
    args = Namespace(
        # Data and path information
        frequency_cutoff=25,
        model_name="birnn",
        model_state_file='birnn',
        train_csv='data/train.csv',
        save_dir='saved_models/multilabel/birnn/',
        vectorizer_file='vectorizer.json',
        weight_decay=0.,
        train_batch_size=16,
        validation_batch_size=256,
        early_stopping_criteria=5,
        learning_rate=0.0005,
        num_epochs=100,
        seed=1337,
        embed_dim=100,
        num_hidden=100,
        num_layers=2,
        save_every_epoch_n=5
    )
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def get_bert_args():
    args = Namespace(
        # Data and path information
        frequency_cutoff=25,
        model_name="birnn",
        model_state_file='birnn',
        train_csv='data/train.csv',
        save_dir='saved_models/multilabel/birnn/',
        vectorizer_file='vectorizer.json',
        weight_decay=0.,
        train_batch_size=16,
        validation_batch_size=256,
        early_stopping_criteria=5,
        seq_len=24,
        learning_rate=3e-5,
        num_epochs=100,
        seed=1337,
        save_every_epoch_n=5
    )
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args