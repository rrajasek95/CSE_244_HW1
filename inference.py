import torch
import os
import config
from torch_models.multiclass.mlp import MultilayerPerceptron
from torch_models.multilabel.mlp_multilabel import MultiLabelMLP
from torch_models.multilabel.deep_fc import DeepFC
from torch_models.multilabel.cnn import OneHotCNN
from torch_models.multilabel.lstm import BiRNN

import model_load

import pandas as pd
from pprint import pprint

from collections import Counter

def save_predictions_to_file(file, predictions):
    mlp_df = pd.DataFrame({'CORE RELATIONS': predictions})
    mlp_df.to_csv(file, index_label='ID')

def get_test_utterances():
    test_data = pd.read_csv('data/hw1_test.csv')
    utterances = test_data.UTTERANCE.tolist()

    return utterances

def get_single_label_predictions(model, vectorizer, utterances):
    predictions = []

    for utterance in utterances:
        vectorized_utt = vectorizer.input_vectorizer.vectorize(utterance)
        vectorized_utt = torch.tensor(vectorized_utt).view(1, -1)
        result = model(vectorized_utt, apply_softmax=True)

        argmax = result.argmax(dim=1).item()
        relation = vectorizer.output_vectorizer.vocabulary.lookup_index(argmax)

        predictions.append(relation)
    return predictions

def get_multilabel_predictions(model, vectorizer, utterances):
    predictions = []

    for utterance in utterances:
        vectorized_utt = vectorizer.input_vectorizer.vectorize(utterance)
        vectorized_utt = torch.tensor(vectorized_utt).unsqueeze(0)
        result = torch.sigmoid(model(vectorized_utt))
        idxs = (result >= 0.5).nonzero().tolist()

        rels = []
        if len(idxs) == 0:
            argmax = result.argmax(dim=1).item()
            label = vectorizer.output_vectorizer.vocabulary.lookup_index(argmax)
            predictions.append(label)
        else:
            for idx in idxs:
                relation = vectorizer.output_vectorizer.vocabulary.lookup_index(idx[1])
                rels.append(relation)
            predictions.append(" ".join(rels))
    return predictions

def mlp_inference():
    args = config.get_mlp_args()
    vectorizer = model_load.load_vectorizer(model_load.get_vectorizer_path(args))
    model = MultilayerPerceptron(
        input_dim=len(vectorizer.input_vectorizer.vocabulary),
        hidden_dim=args.hidden_dim,
        output_dim=len(vectorizer.output_vectorizer.vocabulary))
    model_load.load_best_model(model, args.save_dir, args.model_name)
    utts = get_test_utterances()

    predictions = get_single_label_predictions(model, vectorizer, utts)
    save_predictions_to_file("submissions/mlp_inference.csv", predictions)
    
def mlp_multilabel_inference():
    args = config.get_mlp_args()
    model_load.load_mlp(args)
    utts = get_test_utterances()

    predictions = get_multilabel_predictions(model, vectorizer, utts)
    save_predictions_to_file("submissions/mlp_multilabel_inference.csv", predictions)

def deep_fc_multilabel_inference():
    args = config.get_deep_fc_args()
    model_load.load_deepfc(args)
    utts = get_test_utterances()
    predictions = get_multilabel_predictions(model, vectorizer, utts)
    save_predictions_to_file("submissions/deep_fc_inference.csv", predictions)

def onehot_cnn_inference():
    args = config.get_onehot_cnn_args()
    vectorizer = model_load.load_vectorizer(model_load.get_vectorizer_path(args))
    model = OneHotCNN(
        initial_num_channels=len(vectorizer.input_vectorizer.vocabulary),
        num_channels=args.num_channels,
        output_dim=len(vectorizer.output_vectorizer.vocabulary)) 
    model_load.load_best_model(model, args.save_dir, args.model_name)
    utts = get_test_utterances()

    predictions = get_multilabel_predictions(model, vectorizer, utts)
    save_predictions_to_file("submissions/onehot_cnn_inference.csv", predictions)

def birnn_inference():
    args = config.get_birnn_args()
    model_load.load_birnn(args)
    utts = get_test_utterances()

    predictions = get_multilabel_predictions(model, vectorizer, utts)
    save_predictions_to_file("submissions/birnn_inference.csv", predictions)

def kfold_rnn_inference():
    args = config.get_birnn_args()

    model_wise_predictions = []

    for i in range(5):
        args = config.get_birnn_args()
        args.model_name += str(i)
        args.model_state_file += str(i)
        vectorizer = model_load.load_vectorizer(os.path.join(args.save_dir, "vectorizers_{}.pth".format(i)))
        model = BiRNN(
        vocab_size=len(vectorizer.input_vectorizer.vocabulary),
        embed_dim=args.embed_dim,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        output_dim=len(vectorizer.output_vectorizer.vocabulary))

        model_load.load_best_model(model, args.save_dir, args.model_name)
        utts = get_test_utterances()

        predictions = get_multilabel_predictions(model, vectorizer, utts)
        model_wise_predictions.append(predictions)

    pprint(model_wise_predictions)
    num_test = len(model_wise_predictions[0])

    final_predictions = []

    for i in range(num_test):
        prediction_counter = Counter()
        for j in range(5):
            instance_prediction = model_wise_predictions[j][i]
            prediction_counter.update(instance_prediction.split(" "))
        # 
        majority_classes = [ cls for (cls, count) in prediction_counter.items() if count > 2]
        if len(majority_classes) == 0:
            print("Majority issue")
            cls, ct = prediction_counter.most_common(1)[0]
            final_predictions.append(cls)
        else:
            final_predictions.append(" ".join(majority_classes))

    save_predictions_to_file("submissions/kfold_birnn_inference.csv", final_predictions)


def kfold_birnn_inference_averaged():
    
    models = []
    vectorizers = []

    for i in range(5):
        args = config.get_birnn_args()
        args.model_name += str(i)
        args.model_state_file += str(i)
        vectorizer = model_load.load_vectorizer(os.path.join(args.save_dir, "vectorizers_{}.pth".format(i)))
        model = BiRNN(
        vocab_size=len(vectorizer.input_vectorizer.vocabulary),
        embed_dim=args.embed_dim,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        output_dim=len(vectorizer.output_vectorizer.vocabulary))

        model_load.load_best_model(model, args.save_dir, args.model_name)

        models.append(model)
        vectorizers.append(vectorizer)

    utterances = get_test_utterances()

    predictions = []

    for utterance in utterances:
        total_result = torch.zeros(46)
        for i in range(5):
            vectorized_utt = vectorizers[i].input_vectorizer.vectorize(utterance)
            vectorized_utt = torch.tensor(vectorized_utt).unsqueeze(0)
            result = torch.sigmoid(models[i](vectorized_utt))
            total_result = total_result + result
        total_result = total_result / 5
        idxs = (total_result >= 0.5).nonzero().tolist()

        rels = []
        if len(idxs) == 0:
            argmax = total_result.argmax(dim=1).item()
            label = vectorizer.output_vectorizer.vocabulary.lookup_index(argmax)
            predictions.append(label)
        else:
            for idx in idxs:
                relation = vectorizer.output_vectorizer.vocabulary.lookup_index(idx[1])
                rels.append(relation)
            predictions.append(" ".join(rels))


    save_predictions_to_file("submissions/kfold_birnn_average_inference.csv", predictions)

if __name__ == '__main__':
    mlp_multilabel_inference()