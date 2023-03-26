import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
# Whatever other imports you need
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt

# You can implement classes and helper functions here too.

class MyDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


class Model(nn.Module):
    def __init__(self, input_dims, output_dims):
        """ 
        """
        super().__init__()

        self.input_layer = nn.Linear(input_dims, 5)
        self.hidden = nn.Linear(5,5)
        self.output = nn.Linear(5,output_dims)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ is called when the model object is called: e.g. model(sample_input)
            - defines, how the input is processed with the previuosly defined layers 
        """
        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.output(x)
        
        return self.softmax(x)



def train(model, train_data):
    dataloader = DataLoader(train_data,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=lambda x: x)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(20):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            model_input = torch.stack([sample[0] for sample in batch])
            ground_truth = torch.Tensor([sample[1] for sample in batch])

            # send your batch of sentences to the forward function of the model
            output = model(model_input)

            # compare the output of the model to the ground truth to calculate the loss
            # the lower the loss, the closer the model's output is to the ground truth
            loss = loss_function(output, ground_truth.long())


            # print average loss for the epoch
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')

            # train the model based on the loss:
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # reset gradients
            optimizer.zero_grad()
        print()


def test(model, test_X, test_y):
    y_pred = [int(dist.argmax()) for dist in model(test_X)]
    matrix = pd.DataFrame(confusion_matrix(test_y_t,y_pred))
    # matrix.index = classes
    # matrix.columns = classes
    return matrix



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    # implement everything you need here
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
    model = Model(inputdims, outputdims) 

    train(model, train_data)                            
    conf_matrix = test(model, test_X_t, test_y_t)


    print('\nResult:\n')
    print(conf_matrix)
    print('\nwhere columns = true values and rows = model predictions\n')
    for i in range(len(classes)):
        print('{} = {}'.format(i, classes[i]))












