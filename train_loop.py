from tqdm import tqdm
from model_evaluation import compute_batch_accuracy, compute_batch_prec_recall_f1

import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np

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


def train_single_label_model(model, optimizer, datasets, args):
    print("Running training loop for model", args.model_name)

    # For a multiclass classification problem, the loss is crossentropyloss
    loss_func = nn.CrossEntropyLoss().to(args.device)

    best_validation_acc = 0.0
    best_validation_model_params = None
    best_model_path = os.path.join(args.save_dir, args.model_name + "_best.pth")

    for epoch_index in range(args.num_epochs):
        train_batch_generator = generate_batches(datasets.train_dataset,
                                           batch_size=args.train_batch_size,
                                           device=args.device)
        print("Epoch {}:".format(epoch_index))
        running_loss = 0.0
        running_accuracy = 0.0
        model.train()

        for batch_index, batch_dict in tqdm(enumerate(train_batch_generator)):
            # Set the gradient to 0
            optimizer.zero_grad()
            y_pred = model(x_in=batch_dict['x_data'])
            loss = loss_func(y_pred, batch_dict['y_target'])

            # Backpropagate the loss
            loss.backward() 

            # Compute loss
            batch_loss = loss.item()
            running_loss += (batch_loss - running_loss) / (batch_index + 1)

            # Compute accuracy
            batch_accuracy = compute_batch_accuracy(y_pred, batch_dict['y_target'])
            running_accuracy += (batch_accuracy - running_accuracy) / (batch_index + 1)

            # Perform weight update
            optimizer.step()
            
        print("Minibatch Training Loss: {:.3f}, Running Loss: {:.3f}".format(batch_loss, running_loss))
        print("Minibatch Training Accuracy: {:.2f}, Running Accuracy: {:.2f}".format(batch_accuracy, running_accuracy))

        validation_batch_generator = generate_batches(datasets.validation_dataset,
                                           batch_size=args.validation_batch_size,
                                           device=args.device)
        validation_loss = 0.0
        validation_accuracy = 0.0

        model.eval()
        for batch_index, batch_dict in tqdm(enumerate(validation_batch_generator)):
            y_pred = model(x_in=batch_dict['x_data'])
            loss = loss_func(y_pred, batch_dict['y_target'])

            # Compute loss
            batch_loss = loss.item()
            validation_loss += (batch_loss - running_loss) / (batch_index + 1)

            # Compute accuracy 
            batch_accuracy = compute_batch_accuracy(y_pred, batch_dict['y_target'])
            validation_accuracy += (batch_accuracy - validation_accuracy) / (batch_index + 1)
    
        print("Validation Loss: {:.3f}".format(validation_loss))
        print("Validation Accuracy: {:2f}".format(validation_accuracy))

        # It'll be convenient to track the best validation model and save that
        if best_validation_acc < validation_accuracy:
            best_validation_acc = validation_accuracy
            torch.save(model.state_dict(), best_model_path)
        # Apart from that, I will save the model every n epochs
        if epoch_index % args.save_every_epoch_n == 0:
            epoch_model_path = os.path.join(args.save_dir, "{}_{}.pth".format(args.model_name, epoch_index))
            torch.save(model.state_dict(), epoch_model_path)
        print()

    batch_generator = generate_batches(datasets.test_dataset,
                                       batch_size=len(datasets.test_dataset),
                                       device=args.device)
    test_loss = 0.
    test_accuracy = 0.
    model.eval()
    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = model(x_in=batch_dict['x_data'])
        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'])
        batch_loss = loss.item()
        running_loss += (batch_loss - test_loss) / (batch_index + 1)
        
        # Compute accuracy 
        batch_accuracy = compute_batch_accuracy(y_pred, batch_dict['y_target'])
        test_accuracy += (batch_accuracy - test_accuracy) / (batch_index + 1)

    print("Test loss: {:.3f}".format(running_loss))
    print("Test Accuracy: {:.2f}".format(test_accuracy))


def train_multi_label_model(model, optimizer, datasets, args):
    print("Running training loop for model", args.model_name)

    # For a multiclass classification problem, the loss is crossentropyloss
    loss_func = nn.BCEWithLogitsLoss().to(args.device)

    best_validation_f1 = 0.0
    best_validation_model_params = None
    best_model_path = os.path.join(args.save_dir, args.model_name + "_best.pth")

    vocab = datasets.get_output_vocabulary()
    labels = np.array([vocab.lookup_index(i) for i in range(len(vocab))]).reshape(46, 1)

    for epoch_index in range(args.num_epochs):
        train_batch_generator = generate_batches(datasets.train_dataset,
                                           batch_size=args.train_batch_size,
                                           device=args.device)
        print("Epoch {}:".format(epoch_index))
        running_loss = 0.0
        running_precision = torch.zeros(46)
        running_recall = torch.zeros(46)
        running_f1 = torch.zeros(46)

        model.train()

        for batch_index, batch_dict in tqdm(enumerate(train_batch_generator)):
            # Set the gradient to 0
            optimizer.zero_grad()
            y_pred = model(x_in=batch_dict['x_data'])
            loss = loss_func(y_pred, batch_dict['y_target'])

            # Backpropagate the loss
            loss.backward() 

            # Compute loss
            batch_loss = loss.item()
            running_loss += (batch_loss - running_loss) / (batch_index + 1)

            # Compute accuracy
            precision, recall, f1 = compute_batch_prec_recall_f1(y_pred, batch_dict['y_target'])
            running_precision += (precision - running_precision) / (batch_index + 1)
            running_recall += (recall - running_recall) / (batch_index + 1)
            running_f1 += (f1 - running_f1) / (batch_index + 1)

            # Perform weight update
            optimizer.step()
            
        print("Minibatch Training Loss: {:.3f}, Running Loss: {:.3f}".format(batch_loss, running_loss))
        print("Avg Precision: ", running_precision.mean().item())
        print("Avg Recall: ", running_recall.mean().item())
        print("Avg F1: ", running_f1.mean().item())
        validation_batch_generator = generate_batches(datasets.validation_dataset,
                                           batch_size=args.validation_batch_size,
                                           device=args.device)
        validation_loss = 0.0
        validation_precision = torch.zeros(46)
        validation_recall = torch.zeros(46)
        validation_f1 = torch.zeros(46)

        model.eval()
        for batch_index, batch_dict in tqdm(enumerate(validation_batch_generator)):
            y_pred = model(x_in=batch_dict['x_data'])
            loss = loss_func(y_pred, batch_dict['y_target'])

            # Compute loss
            batch_loss = loss.item()
            validation_loss += (batch_loss - validation_loss) / (batch_index + 1)

            # Compute accuracy 
            precision, recall, f1 = compute_batch_prec_recall_f1(y_pred, batch_dict['y_target'])
            validation_precision += (precision - validation_precision) / (batch_index + 1)
            validation_recall += (recall - validation_recall) / (batch_index + 1)
            validation_f1 += (f1 - validation_f1) / (batch_index + 1)
    
        print("Validation Loss: {:.3f}".format(validation_loss))
        print("Validation Precision, Recall, F1:")
        stacked_tensors = torch.stack((validation_precision, validation_recall, validation_f1), 1).numpy()
        prec_recall_F1_df = pd.DataFrame(np.concatenate((labels, stacked_tensors), axis=1), columns = ["Labels","Precision", "Recall", "F1"])
        prec_recall_F1_df.set_index("Labels")
        print(prec_recall_F1_df)
        print("Validation Avg Precision: ", validation_precision.mean().item())
        print("Validation Avg Recall: ", validation_recall.mean().item())
        print("Validation Avg F1: ", validation_f1.mean().item())
        # It'll be convenient to track the best validation model and save that
        avg_f1 = validation_f1.mean().item()
        if best_validation_f1 < avg_f1:
            best_validation_f1 = avg_f1
            torch.save(model.state_dict(), best_model_path)
        # Apart from that, I will save the model every n epochs
        if epoch_index % args.save_every_epoch_n == 0:
            epoch_model_path = os.path.join(args.save_dir, "{}_{}.pth".format(args.model_name, epoch_index))
            torch.save(model.state_dict(), epoch_model_path)
        print()

    batch_generator = generate_batches(datasets.test_dataset,
                                       batch_size=len(datasets.test_dataset),
                                       device=args.device)
    test_loss = 0.
    test_precision = torch.zeros(46)
    test_recall = torch.zeros(46)
    test_f1 = torch.zeros(46)
    model.eval()
    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = model(x_in=batch_dict['x_data'])
        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'])
        batch_loss = loss.item()
        running_loss += (batch_loss - test_loss) / (batch_index + 1)
        
        # Compute accuracy 
        precision, recall, f1 = compute_batch_prec_recall_f1(y_pred, batch_dict['y_target'])
        test_precision += (precision - test_precision) / (batch_index + 1)
        test_recall += (recall - test_recall) / (batch_index + 1)
        test_f1 += (f1 - test_f1) / (batch_index + 1)

    stacked_tensors = torch.stack((test_precision, test_recall, test_f1), 1).numpy()
    prec_recall_F1_df = pd.DataFrame(np.concatenate((labels, stacked_tensors), axis=1), columns = ["Labels","Precision", "Recall", "F1"])
    prec_recall_F1_df.set_index("Labels")
    print("Test loss: {:.3f}".format(running_loss))
    print("Test Precision, Recall, F1:")
    print(prec_recall_F1_df)
    print("Test Avg Precision: ", test_precision.mean().item())
    print("Test Avg Recall: ", test_recall.mean().item())
    print("Test Avg F1: ", test_f1.mean().item())


    