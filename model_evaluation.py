import torch
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np

import config
import model_load
import prep_data
import os

from torch.utils.data import DataLoader

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}

        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def compute_batch_accuracy(y_pred, y_target):
    # Applicable only for multiclass case
    y_target = y_target.cpu()
    
    probs = torch.softmax(y_pred, dim=1)
    
    y_pred_ind = probs.argmax(dim=1).cpu()

    return (y_target == y_pred_ind).sum().float()/len(y_pred)

def compute_batch_prec_recall_f1(y_pred, y_target, average=False):
    y_target = y_target.cpu()
    
    probs = torch.sigmoid(y_pred)
    threshold = (probs >= 0.5).float().cpu()
    true_positives = (y_target * threshold).sum(0)
    total_positives = threshold.sum(0)
    true_totals = y_target.sum(0)
    prec = []
    for (n, d) in zip(true_positives, total_positives):
        if d.item() != 0:
            prec.append((n / d).item())
        else:
            prec.append(0.0)

    prec_tensor = torch.Tensor(prec)
    rec = []

    for (n, d) in zip(true_positives, true_totals):
        if d.item() != 0:
            rec.append((n / d).item())
        else:
            rec.append(0.0)

    rec_tensor = torch.Tensor(rec)

    f1 = []
    for (n, d) in zip(prec, rec):
        if n!= 0 and d != 0:
            f1.append(2 * n * d / (n + d))
        else:
            f1.append(0)

    f1_tensor = torch.tensor(f1)
    return (prec_tensor, rec_tensor, f1_tensor)

def compute_micro_avg_prec_recall_f1(y_pred, y_target):
    y_target = y_target.cpu()
    
    probs = torch.sigmoid(y_pred)
    threshold = (probs >= 0.5).float().cpu()
    true_positives = (y_target * threshold).sum()
    total_positives = threshold.sum()
    true_totals = y_target.sum()

    prec = 0
    rec = 0
    f1 = 0

    if total_positives > 0:
        prec = true_positives / total_positives
    if true_totals > 0:
        rec = true_positives / true_totals

    if prec > 0 and rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    return (prec, rec, f1)

def run_test_step(model, datasets, args):
    print("Running evaluation for model:", args.model_name)
    vocab = datasets.get_output_vocabulary()
    labels = np.array([vocab.lookup_index(i) for i in range(len(vocab))]).reshape(46, 1)
    batch_generator = generate_batches(datasets.test_dataset,
                                       batch_size=len(datasets.test_dataset),
                                       device=args.device)
    model.eval()
    test_precision = torch.zeros(46)
    test_recall = torch.zeros(46)
    test_f1 = torch.zeros(46)

    test_micro_precision = 0
    test_micro_recall = 0
    test_micro_f1 = 0

    for batch_index, batch_dict in enumerate(batch_generator):
        print(batch_dict['x_data'].shape)
        # compute the output
        y_pred = model(x_in=batch_dict['x_data'])
        
        # Compute accuracy 
        precision, recall, f1 = compute_batch_prec_recall_f1(y_pred, batch_dict['y_target'])
        test_precision += (precision - test_precision) / (batch_index + 1)
        test_recall += (recall - test_recall) / (batch_index + 1)
        test_f1 += (f1 - test_f1) / (batch_index + 1)
        micro_prec, micro_rec, micro_f1 = compute_micro_avg_prec_recall_f1(y_pred, batch_dict['y_target'])

        test_micro_precision = (micro_prec - test_micro_precision) / (batch_index + 1)
        test_micro_recall = (micro_rec- test_micro_recall) / (batch_index + 1)
        test_micro_f1 = (micro_f1 - test_micro_f1) / (batch_index + 1)

    stacked_tensors = torch.stack((test_precision, test_recall, test_f1), 1).numpy()
    prec_recall_F1_df = pd.DataFrame(np.concatenate((labels, stacked_tensors), axis=1), columns = ["Labels","Precision", "Recall", "F1"])
    prec_recall_F1_df.set_index("Labels")
    print(prec_recall_F1_df)
    print("Test Avg Precision: ", test_precision.mean().item())
    print("Test Avg Recall: ", test_recall.mean().item())
    print("Test Avg F1: ", test_f1.mean().item())

    print("Test Micro Precision: ", test_micro_precision)
    print("Test Micro Recall: ", test_micro_recall)
    print("Test Micro F1: ", test_micro_f1)

def run_ensemble_test_step(models, datasets, args):
    print("Running evaluation for model:", args.model_name)
    vocab = datasets.get_output_vocabulary()
    labels = np.array([vocab.lookup_index(i) for i in range(len(vocab))]).reshape(46, 1)
    batch_generator = generate_batches(datasets.test_dataset,
                                       batch_size=len(datasets.test_dataset),
                                       device=args.device)
    test_precision = torch.zeros(46)
    test_recall = torch.zeros(46)
    test_f1 = torch.zeros(46)

    test_micro_precision = 0
    test_micro_recall = 0
    test_micro_f1 = 0

    for batch_index, batch_dict in enumerate(batch_generator):
        print(batch_dict['x_data'].shape)
        # average predictions
        idx = 0
        y_pred = torch.zeros(46)
        for model in models:
            y_pred = model(x_in=batch_dict['x_data'])
            idx += 1
        y_pred = y_pred/idx
        
        # Compute accuracy 
        precision, recall, f1 = compute_batch_prec_recall_f1(y_pred, batch_dict['y_target'])
        test_precision += (precision - test_precision) / (batch_index + 1)
        test_recall += (recall - test_recall) / (batch_index + 1)
        test_f1 += (f1 - test_f1) / (batch_index + 1)
        micro_prec, micro_rec, micro_f1 = compute_micro_avg_prec_recall_f1(y_pred, batch_dict['y_target'])

        test_micro_precision = (micro_prec - test_micro_precision) / (batch_index + 1)
        test_micro_recall = (micro_rec- test_micro_recall) / (batch_index + 1)
        test_micro_f1 = (micro_f1 - test_micro_f1) / (batch_index + 1)

    stacked_tensors = torch.stack((test_precision, test_recall, test_f1), 1).numpy()
    prec_recall_F1_df = pd.DataFrame(np.concatenate((labels, stacked_tensors), axis=1), columns = ["Labels","Precision", "Recall", "F1"])
    prec_recall_F1_df.set_index("Labels")
    print(prec_recall_F1_df)
    print("Test Avg Precision: ", test_precision.mean().item())
    print("Test Avg Recall: ", test_recall.mean().item())
    print("Test Avg F1: ", test_f1.mean().item())

    print("Test Micro Precision: ", test_micro_precision)
    print("Test Micro Recall: ", test_micro_recall)
    print("Test Micro F1: ", test_micro_f1)

def load_train_df(args):
    return pd.read_csv(args.train_csv)

def run_test_perceptron():
    args = config.get_perceptron_args()
    vectorizer, model = model_load.load_perceptron(args)

    train_df = load_train_df(args)
    datasets = prep_data.SingleLabelDatasets.from_train_dataframe(train_df)

    run_single_test_step(model, datasets, args)

def run_test_mlp_multilabel():
    args = config.get_mlp_multilabel_args()
    args.device="cpu"
    vectorizer, model = model_load.load_mlp(args)

    train_df = load_train_df(args)
    datasets = prep_data.OneHotMultilabelDatasets.from_train_dataframe(train_df)
    datasets.set_vectorizer(vectorizer)

    run_test_step(model, datasets, args)

def run_test_deepfc():
    args = config.get_deep_fc_args()
    args.device="cpu"
    vectorizer, model = model_load.load_deepfc(args)

    train_df = load_train_df(args)
    datasets = prep_data.OneHotMultilabelDatasets.from_train_dataframe(train_df)
    datasets.set_vectorizer(vectorizer)

    run_test_step(model, datasets, args)

def run_test_cnn():
    args = config.get_onehot_cnn_args()
    args.device="cpu"
    vectorizer, model = model_load.load_onehot_cnn(args)

    train_df = load_train_df(args)
    datasets = prep_data.OneHotSequenceMultilabelDatasets.from_train_dataframe(train_df)
    datasets.set_vectorizer(vectorizer)

    run_test_step(model, datasets, args)


def run_birnn():
    args = config.get_birnn_args()
    args.device="cpu"
    vectorizer, model = model_load.load_birnn(args)

    train_df = load_train_df(args)
    datasets = prep_data.OneHotSequenceMultilabelDatasets.from_train_dataframe(train_df)
    datasets.set_vectorizer(vectorizer)

    run_test_step(model, datasets, args)

def run_kfold_birnn():
    models = []
    vectorizers = []

    for i in range(5):
        args = config.get_birnn_args()
        args.model_name += str(i)
        args.model_state_file += str(i)
        args.device="cpu"
        vectorizer, model = model_load.load_birnn(args)

        models.append(model)
        vectorizers.append(vectorizer)

    train_df = load_train_df(args)
    datasets = prep_data.EmbeddedSequenceMultilabelDatasets.from_train_dataframe(train_df,dim=300)

    run_ensemble_test_step(models, datasets, args)

if __name__ == '__main__':
    run_kfold_birnn()
