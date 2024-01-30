"""
Build and train tiny example PyTorch model to be used in PyT_petsc_test.py
partly lifted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
and https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html .
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html is 
also useful.
"""

import torch
import torch.nn as nn

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

#######################
# 1D U-Net components #
#######################
    
# taken partly from 
# https://pyimagesearch.com/2023/11/06/image-segmentation-with-u-net-in-pytorch-the-grand-finale-of-the-autoencoder-series/

# straight convolution block, pot. with change in depth (num of channels)
class DualConv(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(DualConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_ch, output_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(output_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_ch, output_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(output_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_block(x)
    
# restriction (encoding)
class Contract(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(Contract, self).__init__()
        #doing average pooling instead of max pooling because we want to smooth over the
        #field instead of picking 'most important feature'
        #kernel size = stride = 2, so grid is coarsened by factor of 2 each time.
        self.down_conv = nn.Sequential(nn.AvgPool1d(2), DualConv(input_ch, output_ch))
    def forward(self, x):
        return self.down_conv(x)

# prolongation (decoding)
class Expand(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(Expand, self).__init__()
        self.up = nn.ConvTranspose1d(input_ch, input_ch // 2, kernel_size=2, stride=2)
        self.conv = DualConv(input_ch, output_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print(x1.shape)
        #x2 is the tensor from the restriction step
        diff = x2.size()[2] - x1.size()[2]
        x1 = nn.functional.pad(
            x1, [diff // 2, diff - diff // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#putting it together:
class OneD_UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(OneD_UNet, self).__init__()
        self.initial = DualConv(input_channels, 16)
        self.down1 = Contract(16, 32)
        self.down2 = Contract(32, 64)
        self.down3 = Contract(64, 128)
        self.up2 = Expand(128, 64)
        self.up3 = Expand(64, 32)
        self.up4 = Expand(32, 16)
        self.final = DualConv(16, output_channels)
    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.final(x)
        return out



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
    UNet = OneD_UNet(1, 1).to(device)


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


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            if model == UNet:
                # Reshape the data to (batch_size, channels, sequence_length)
                X = X.unsqueeze(0)
                X = X.view(100, 1, 100)
                y = y.unsqueeze(0)
                y = y.view(100, 1, 100)
            
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
                if model == UNet:
                    # Reshape the data to (batch_size, channels, sequence_length)
                    X = X.unsqueeze(0)
                    X = X.view(100, 1, 100)
                    y = y.unsqueeze(0)
                    y = y.view(100, 1, 100)
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")
        wandb.log({"Test Loss":test_loss})

    '''
    # train dense network
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


    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, tinymodel, loss_fn, optimizer)
        test(validation_loader, tinymodel, loss_fn)
        
    print("Done!")

    torch.save(tinymodel.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/poisson.pth")
    print("Saved PyTorch Model State to poisson.pth")

    wandb.finish()
    '''

    # train U-Net
    optimizer = torch.optim.SGD(UNet.parameters(), lr=0.06, momentum=0.9)
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
        "architecture": "UNet",
        "dataset": "randompoisson_100by100",
        "epochs": 10,
        }
    )
    
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
    
        train(training_loader, UNet, loss_fn, optimizer)
        test(validation_loader, UNet, loss_fn)
        
    print("Done!")

    torch.save(UNet.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/unet_poisson.pth")
    print("Saved PyTorch Model State to unet_poisson.pth")

    #wandb.finish()



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
    #helmholtztrain()
    