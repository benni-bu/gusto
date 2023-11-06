"""
Build and train tiny example PyTorch model to be used in PtY_petsc_test.py
partly lifted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
and https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb
import numpy as np

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 100)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

tinymodel = TinyModel()


#generate some training data (solving a simple tridiagonal linear system)
matrix = np.eye(10) + np.eye(10, k=1) + np.eye(10, k=-1)
#generate 1000 input vectors of size 10, multiply them with matrix to get output vectors
vec_in = np.random.rand(1000, 10)
vec_out = np.dot(vec_in, matrix)

# Convert data to PyTorch tensors
input_tensors = torch.tensor(vec_in, dtype=torch.float32)
output_tensors = torch.tensor(vec_out, dtype=torch.float32)

# Create a TensorDataset from input and output tensors
dataset = TensorDataset(input_tensors, output_tensors)

# Define the sizes for training and test sets
train_size = int(0.8 * len(dataset))  # 80% of the data for training
validation_size = len(dataset) - train_size  # Remaining 20% for testing

# Use random_split to create training and test sets
training_set, validation_set = random_split(dataset, [train_size, validation_size])


# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=4, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="ML_acc_solvers",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "dense",
    "dataset": "simplelinalg",
    "epochs": 10,
    }
)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
