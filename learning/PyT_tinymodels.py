"""
Build and train tiny example PyTorch model to be used in PyT_petsc_test.py
partly lifted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
and https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html .
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html is 
also useful.
"""

import torch
from torchvision import models

#define the model architecture
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
    
class TinyCNN(torch.nn.Module):

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


def poissontrain():
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import wandb
    import numpy as np
    import matplotlib.pyplot as plt

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    tinymodel = TinyModel().to(device)

    #generate some training data (solving a 1D discrete Poisson-type linear system)
    matrix = 2*np.eye(100) - np.eye(100, k=-1) - np.eye(100, k=1)
    plt.imshow(matrix)
    plt.show()
    #generate 1000 vectors of size 10, multiply them with matrix to get RHS vectors.
    #We want our network to learn the mapping from the RHS vector to the LHS vector
    #in a system Ax=b. Here, x:=vec_out, b:=vec_in
    vec_out = np.random.rand(1000, 100)
    vec_in = np.dot(vec_out, matrix)

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
    training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=100, shuffle=False)


    optimizer = torch.optim.SGD(tinymodel.parameters(), lr=0.06, momentum=0.9)
    loss_fn = torch.nn.MSELoss()


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.06,
        "layers": 2,
        "optim": "SGD",
        "architecture": "dense",
        "dataset": "randompoisson_100by100",
        "epochs": 10,
        }
    )


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            wandb.log({"Training Loss":loss})


    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")
        wandb.log({"Test Loss":test_loss})


    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, tinymodel, loss_fn, optimizer)
        test(validation_loader, tinymodel, loss_fn)
        
    print("Done!")

    torch.save(tinymodel.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/poisson.pth")
    print("Saved PyTorch Model State to poisson.pth")

    wandb.finish()


def helmholtztrain():
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import wandb
    import numpy as np
    import matplotlib.pyplot as plt

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    tinymodel = TinyModel().to(device)

    #generate some training data (solving a 1D discrete inhomogeneous
    # Helmholtz-type linear system).
    #The equation we set up is of the form (\nabla^2 + 1)x = b
    #i.e., wavenumber is 1.
    matrix = 2*np.eye(100) + np.eye(100) - np.eye(100, k=-1) - np.eye(100, k=1)
    plt.imshow(matrix)
    plt.show()
    #generate 1000 vectors of size 10, multiply them with matrix to get RHS vectors.
    #We want our network to learn the mapping from the RHS vector to the LHS vector
    #in a system Ax=b. Here, x:=vec_out, b:=vec_in
    vec_out = np.random.rand(1000, 100)
    vec_in = np.dot(vec_out, matrix)

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
    training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=100, shuffle=False)


    optimizer = torch.optim.SGD(tinymodel.parameters(), lr=0.06, momentum=0.9)
    loss_fn = torch.nn.MSELoss()


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.06,
        "layers": 2,
        "optim": "SGD",
        "architecture": "dense",
        "dataset": "randomhelmholtz_100by100",
        "epochs": 10,
        }
    )


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            wandb.log({"Training Loss":loss})


    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")
        wandb.log({"Test Loss":test_loss})


    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, tinymodel, loss_fn, optimizer)
        test(validation_loader, tinymodel, loss_fn)
        
    print("Done!")

    torch.save(tinymodel.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/helmholtz.pth")
    print("Saved PyTorch Model State to helmholtz.pth")

    wandb.finish()


#avoid running the rest of the script when just importing ML model from elsewhere
if __name__ == '__main__':
    poissontrain()
    helmholtztrain()
    