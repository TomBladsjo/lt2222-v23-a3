import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix


class MyDataset(Dataset):
    def __init__(self, X, y):
        """Stores documents and labels as X and y.
        """
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


class Model(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_layer=False, nonlinear=False):
        """ Args:
                input_dims: The dimensions of the input data (integer).
                output_dims: The dimensions of the output (integer). Should be 
                    the same as the number of classes. 
                hidden_layer: If not false, adds a hidden layer. If provided as an 
                integer, this will be the dimensions of the hidden layer. If simply
                set to true, the dimensions of the hidden layer will be the mean of
                the input and output dimensions. 
                nonlinear: The type of nonlinear activation function between the model 
                layers. The options are 'relu' for ReLU and 'tanh' for Tanh. If false,
                no activation function is used. 
        """
        super().__init__()
        self.hidden_layer = hidden_layer
        self.nonlinear = nonlinear
        inside_dims = hidden_layer if type(hidden_layer) == int else (input_dims+output_dims)//2

        self.input = nn.Linear(input_dims, inside_dims)
        self.hidden = nn.Linear(inside_dims,inside_dims)
        self.relu = nn.functional.relu
        self.tanh = nn.Tanh()
        self.output = nn.Linear(inside_dims,output_dims)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x) if self.nonlinear == 'relu' else x
        x = self.tanh(x) if self.nonlinear == 'tanh' else x
        x = self.hidden(x) if self.hidden_layer else x
        x = self.relu(x) if self.nonlinear == 'relu' else x
        x = self.tanh(x) if self.nonlinear == 'tanh' else x
        x = self.output(x)
        
        return self.softmax(x)




def train(model, train_data, epochs):
    """Given a model and training data trains the model for the specified number of epochs.
    """
    dataloader = DataLoader(train_data,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=lambda x: x)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            model_input = torch.stack([sample[0] for sample in batch])
            ground_truth = torch.Tensor([sample[1] for sample in batch])

            output = model(model_input)
            loss = loss_function(output, ground_truth.long())
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print()


def test(model, test_X, test_y):
    """Takes a model, a set of test data and the corresponding ground truth labels.
    Returns a confusion matrix where the columns are the true values and the rows
    are the model predictions.
    """
    y_pred = [int(dist.argmax()) for dist in model(test_X)]
    matrix = pd.DataFrame(confusion_matrix(test_y_t,y_pred))
    return matrix



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--hidden", "-H", dest="hidden_layer", type=int, nargs='?', const=True, help="The size (integer) of the hidden layer")
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
    
    model = Model(inputdims, outputdims, hidden_layer=args.hidden_layer, nonlinear=args.nonlinear)  
    
    print("Training model...")
    train(model, train_data, args.epochs)       
    print("Done training!")
    
    conf_matrix = test(model, test_X_t, test_y_t)
    print('\nResult:\n')
    print(conf_matrix)
    print('\nwhere columns = true values and rows = model predictions\n')
    for i in range(len(classes)):
        print('{} = {}'.format(i, classes[i]))













