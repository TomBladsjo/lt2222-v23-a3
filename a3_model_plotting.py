import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

from a3_model import MyDataset, Model, train


def test_settings(training_data, test_X, test_y, inputdims, outputdims, epochs=10,
                    nonlinear=False, start=5, stop=200, step=10):
    precision = []
    recall = []
    sizes = list(range(start, stop, step))
    for size in sizes:
        model = Model(inputdims, outputdims, hidden_layer=size, nonlinear=nonlinear)
        print("Training model with hidden layer size {}...".format(size))
        train(model, training_data, epochs)
        print("Testing model...")
        outputs = model(test_X_t)
        y_pred = pd.Series(outputs.argmax(dim=1).numpy())
        precision.append(precision_score(test_y, y_pred, average='weighted', zero_division=0))
        recall.append(recall_score(test_y, y_pred, average='weighted', zero_division=0))
    
    return sizes, precision, recall




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model with different hidden layer sizes, write results to plot.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("outfile", type=str, help="The output file for the plot (png).")
    parser.add_argument("--start", dest="start", type=int, default=5, help="The smallest hidden layer size to test (default=5).")
    parser.add_argument("--stop", dest="stop", type=int, default=200, help="The largest hidden layer size to test (default=200).")
    parser.add_argument("--step", "-st", dest="step", type=int, default=10, help="The step size (integer) when testing different layer sizes (default=10)")
    parser.add_argument("--nonlinearity", "-nl", dest="nonlinear", type=str, default=False, help="The nonlinear activation function (string). Options are 'tanh' or 'relu'.")
    parser.add_argument("--epochs", "-E", dest="epochs", type=int, default=10, help="The number of training epochs (integer).")
    
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    data = pd.read_csv(args.featurefile, index_col=['vectors'])

    
    classes = list(data['class'].unique())
    trainset = data[data['split']=='train']
    testset = data.drop(trainset.index, axis=0)

    train_X = trainset.drop(['class','split', 'class_id'], axis=1)
    train_y = trainset['class_id']
    test_X = testset.drop(['class','split', 'class_id'], axis=1)
    test_y = testset['class_id']

    train_X_t = torch.Tensor(train_X.to_numpy())
    train_y_t = torch.LongTensor(train_y.to_numpy())
    test_X_t = torch.Tensor(test_X.to_numpy())
    test_y_t = torch.LongTensor(test_y.to_numpy())

    train_data = MyDataset(train_X_t, train_y_t)
    inputdims = train_X_t.shape[1]
    outputdims = len(classes)
    
    
    sizes, precision, recall = test_settings(train_data, test_X_t, test_y_t, inputdims, outputdims, epochs=args.epochs,
                    nonlinear=args.nonlinear, start=args.start, stop=args.stop, step=args.step)

    print("Creating plot...")
    
    fig, ax = plt.subplots()
    ax.set(ylim=(0, 1))
    plt.plot(sizes, precision, color='orange', label='Precision')
    plt.plot(sizes, recall, color='blue', label='Recall')
    plt.legend()
    plt.xlabel("Hidden layer dimensions")
    plt.title("The effect of hidden layer size on precision and recall")
    print("Saving as {}...".format(args.outfile))
    print("All done!")
    plt.savefig(args.outfile)














