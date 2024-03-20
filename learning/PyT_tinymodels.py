"""
Build and train example PyTorch models to be used in PyT_petsc_test.py and the 
one-dimensional problem in petsc_toyproblem.py.
partly lifted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
and https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html .
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html is 
also useful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from sklearn.preprocessing import MinMaxScaler

###########################
# Dense, one hidden Layer #
###########################

class Dense(nn.Module):

    def __init__(self):
        super(Dense, self).__init__()

        self.linear1 = nn.Linear(100, 200)
        self.activation = nn.ELU()
        self.linear2 = nn.Linear(200, 100)
        #add a convolutional layer to hopefully smooth the output
        #self.conv = nn.Conv1d(1,1,kernel_size=5,stride=1,padding=2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

#the fully connected NN (above) produces very noisy outputs, so we define a simple smoother to pass over them:
class smoother(nn.Module):
    def __init__(self):
        super(smoother, self).__init__()
        self.sm_kernel = torch.tensor([0.05, 0.075, 0.1, 0.175, 0.2, 0.175, 0.1, 0.075, 0.05], dtype=torch.float32)

    def forward(self, x):
        output = x.unsqueeze(0)
        output = x.unsqueeze(0)
        output = F.conv1d(output, self.sm_kernel.view(1, 1, -1), padding=4)
        output = F.conv1d(output, self.sm_kernel.view(1, 1, -1), padding=4)
        output = output.squeeze()
        return output
    
#####################
# linear regression #
#####################
    
#once as a fully connected network
class LinReg(nn.Module):

    def __init__(self):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(100, 100)

    def forward(self, x):
        x = self.linear(x)
        return x
    

# and once as a linear convolution (essentially Ackmann et al.)
class LinConv(nn.Module):

    def __init__(self):
        super(LinConv, self).__init__()
        self.conv = nn.Conv1d(1,1,kernel_size=5,stride=1,padding=2)

    def forward(self, x):
        x = self.conv(x)
        return x


#######################
# 1D U-Net components #
#######################
    
# taken partly from 
# https://pyimagesearch.com/2023/11/06/image-segmentation-with-u-net-in-pytorch-the-grand-finale-of-the-autoencoder-series/
# and adapted according to the first U-Net in Azulay and Treister (2022)

# ResNet block
class ResConv(nn.Module):
    def __init__(self, num_ch):
        super(ResConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_ch, num_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(num_ch),
            nn.ELU(inplace=True),
        )
        self.conv = nn.Conv1d(num_ch, num_ch, 3, padding=1, bias=False)
        self.actbatch = nn.Sequential(
            nn.BatchNorm1d(num_ch),
            nn.ELU(inplace=True),
        )
    def forward(self, x):
        out = self.conv_block(x)
        out = x + self.conv(out)
        out = self.actbatch(out)
        return out
    
# restriction (encoding)
class Contract(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(Contract, self).__init__()
        #doing strided convolution instead of pooling because we want to learn the
        #transfer operators between different 'grids'
        #stride = 2, so grid is coarsened by factor of 2 each time.
        self.down_conv = nn.Sequential(
            nn.Conv1d(input_ch, output_ch, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(output_ch),
            nn.ELU(inplace=True),
            ResConv(output_ch),
        )
    def forward(self, x):
        return self.down_conv(x)

# prolongation (decoding)
class Expand(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(Expand, self).__init__()
        self.up = nn.ConvTranspose1d(input_ch, input_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(input_ch, output_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(output_ch),
            nn.ELU(inplace=True),
        )
    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print(x1.shape)
        #x2 is the tensor from the restriction step
        diff = x2.size()[-1] - x1.size()[-1]
        x1 = nn.functional.pad(
            x1, [diff // 2, diff - diff // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#putting it together:
class OneD_UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(OneD_UNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU(inplace=True),
        )
        self.down1 = Contract(16, 32)
        self.down2 = Contract(32, 64)
        self.down3 = Contract(64, 128)
        self.coarse = ResConv(128)
        self.up2 = Expand(128, 64)
        self.up3 = Expand(64, 32)
        self.up4 = Expand(32, 16)
        self.final = nn.Sequential(
            nn.Conv1d(16, output_channels, 3, padding=1),
            nn.BatchNorm1d(output_channels),
            nn.ELU(inplace=True),
        )
    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.coarse(x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.final(x)
        return out


#---------------------------------------------------------------------------------------------------#

                                            ############
                                            # training #
                                            ############
    
#---------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
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

    dense = Dense().to(device)
    UNet = OneD_UNet(1, 1).to(device)
    linreg = LinReg().to(device)
    linconv = LinConv().to(device)    


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        if model == UNet or model == linconv:
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

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
        wandb.log({"Training Loss":loss})


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            if model == UNet or model ==linconv:
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

#define a new loss function that also measures how far off the operator times 
# the output is - this should better reflect what happens with the model 
# output within the solver
class MSEplusOpLoss(nn.Module):
    def __init__(self, op):
        super(MSEplusOpLoss, self).__init__()
        self.op = torch.Tensor(op)
        self.op.unsqueeze(0)
    def forward(self, input, target):
        mse = torch.nn.MSELoss()
        MSE = mse(input, target)
        Ap = torch.matmul(self.op, input)
        Ax = torch.matmul(self.op, target)
        OpNorm = mse(Ap, Ax) * 1e-7
        return MSE + OpNorm


    
def poissontrain():
    #generate some training data (solving a 1D discrete Poisson-type linear system)
    matrix = (2*np.eye(102) - np.eye(102, k=-1) - np.eye(102, k=1))*101**2
    #plt.imshow(matrix)
    #plt.show()

    #generate 1000 vectors of size 100, multiply them with matrix to get RHS vectors.
    #We want our network to learn the mapping from the RHS vector to the LHS vector
    #in a system Ax=b. Here, x:=vec_out, b:=vec_in
    #generate x vectors that are smooth and fixed to zero at the boundaries 
    #to maintain physicality and allow the network to learn.
    xs = np.arange(102, step=1)
    a = 0.1 * np.random.randn(10000)
    b = 0.05 * np.random.randn(10000)
    c = 0.03 * np.random.randn(10000)
    d = 0.03 * np.random.randn(10000)
    vec_out = np.zeros((10000, 102))
    for i in np.arange(10000):
        vec_out[i] = (a[i] * np.sin(np.pi/100 * xs) + b[i] * np.sin(np.pi/100 * 2 * xs) + 
                c[i] * np.sin(np.pi/100 * 3 * xs) + d[i] * np.sin(np.pi/100 * 4 * xs))
    #random x vectors (not recommended, doesn't train well):
    #vec_out = np.random.rand(1000, 100)
        
    #large set ot vectors of ones as a sanity check (create 'perfect preconditioner' for this particular problem)
    #for i in np.arange(1000):
    #    vec_out[i] = (- xs **2 + 100*xs)*0.00005

    #compute RHS based on x vector
    vec_in = np.dot(vec_out, matrix)

    #cut off boundaries because they seem to behave weirdly (bcs not properly specified)
    vec_in = vec_in[:, 1:-1]
    vec_out = vec_out[:, 1:-1]

    #scale by infty-norm of input to make model scale invariant
    norm = np.linalg.norm(vec_in, ord=np.inf, axis = 1, keepdims = True)
    vec_in *= 1/norm
    vec_out *= 1/norm

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
    
    #train linear regression model
    optimizer = torch.optim.Adam(linreg.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        name= "linreg_sm-poisson_inf_100by100",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-3,
        "layers": 0,
        "optim": "Adam",
        "architecture": "dense",
        "dataset": "smooth_poisson_100by100",
        "epochs": 8,
        "activation": "linear"
        }
    )


    epochs = 8
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, linreg, loss_fn, optimizer)
        test(validation_loader, linreg, loss_fn)
        
    print("Done!")

    torch.save(linreg.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/lin_poisson.pth")
    print("Saved PyTorch Model State to lin_poisson.pth")

    wandb.finish()
    
    
    #train linear convolution model
    optimizer = torch.optim.Adam(linconv.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        name= "linconv_sm-poisson_inf_100by100",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-3,
        "layers": 0,
        "optim": "Adam",
        "architecture": "dense",
        "dataset": "smooth_poisson_100by100",
        "epochs": 20,
        "activation": "linear"
        }
    )


    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, linconv, loss_fn, optimizer)
        test(validation_loader, linconv, loss_fn)
        
    print("Done!")

    torch.save(linconv.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/linconv_poisson.pth")
    print("Saved PyTorch Model State to linconv_poisson.pth")

    wandb.finish()


    
    # train dense network
    optimizer = torch.optim.Adam(dense.parameters(), lr=1e-3)
    matrix = (2*np.eye(100) - np.eye(100, k=-1) - np.eye(100, k=1))*100**2
    loss_fn = torch.nn.MSELoss()
    #loss_fn = MSEplusOpLoss(matrix)


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        name= "scaleinf_dense_sm_poisson_100by100",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-3,
        "layers": 2,
        "optim": "Adam",
        "architecture": "dense",
        "dataset": "smoothpoisson_100by100",
        "epochs": 20,
        "activation": "ELU",
        "loss": "MSEplusOpLoss"
        }
    )


    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, dense, loss_fn, optimizer)
        test(validation_loader, dense, loss_fn)
        
    print("Done!")

    torch.save(dense.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/poisson.pth")
    print("Saved PyTorch Model State to poisson.pth")

    wandb.finish()

    
    
    
    # train U-Net
    optimizer = torch.optim.Adam(UNet.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        name= "UNet-ResNet-scaleinf_sm-poisson_100by100",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 5e-3,
        "layers": 19,
        "optim": "Adam",
        "architecture": "Res-UNet",
        "dataset": "smoothpoisson_100by100",
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

    wandb.finish()
    

def helmholtztrain():
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


    optimizer = torch.optim.SGD(dense.parameters(), lr=0.06, momentum=0.9)
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

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, dense, loss_fn, optimizer)
        test(validation_loader, dense, loss_fn)
        
    print("Done!")

    torch.save(dense.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/helmholtz.pth")
    print("Saved PyTorch Model State to helmholtz.pth")

    wandb.finish()


#use weights already trained to learn on new dataset
def transferlearn(in_file, out_file):
    ps = np.loadtxt(out_file, delimiter=',')
    vs = np.loadtxt(in_file, delimiter=',')

    # Convert data to PyTorch tensors
    input_tensors = torch.tensor(vs, dtype=torch.float32)
    output_tensors = torch.tensor(ps, dtype=torch.float32)

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
    
    #train linear regression model (after loading old state dict)
    linreg.load_state_dict(torch.load("/Users/GUSTO/environments/firedrake/src/gusto/learning/lin_poisson.pth"))
    optimizer = torch.optim.Adam(linreg.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        name= "linreg_trans-poisson_inf_100by100",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-3,
        "layers": 0,
        "optim": "Adam",
        "architecture": "dense",
        "dataset": "smooth_poisson_100by100",
        "epochs": 15,
        "activation": "linear"
        }
    )


    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, linreg, loss_fn, optimizer)
        test(validation_loader, linreg, loss_fn)
        
    print("Done!")

    torch.save(linreg.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/lin_poisson_trans.pth")
    print("Saved PyTorch Model State to lin_poisson_trans.pth")

    wandb.finish()
    
    
    #train linear convolution model
    linconv.load_state_dict(torch.load("/Users/GUSTO/environments/firedrake/src/gusto/learning/linconv_poisson.pth"))
    optimizer = torch.optim.Adam(linconv.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        name= "linconv_trans-poisson_inf_100by100",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-3,
        "layers": 0,
        "optim": "Adam",
        "architecture": "dense",
        "dataset": "smooth_poisson_100by100",
        "epochs": 15,
        "activation": "linear"
        }
    )


    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, linconv, loss_fn, optimizer)
        test(validation_loader, linconv, loss_fn)
        
    print("Done!")

    torch.save(linconv.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/linconv_poisson_trans.pth")
    print("Saved PyTorch Model State to linconv_poisson_trans.pth")

    wandb.finish()


    
    # train dense network
    dense.load_state_dict(torch.load("/Users/GUSTO/environments/firedrake/src/gusto/learning/poisson.pth"))
    optimizer = torch.optim.Adam(dense.parameters(), lr=1e-3)
    matrix = (2*np.eye(100) - np.eye(100, k=-1) - np.eye(100, k=1))*100**2
    loss_fn = torch.nn.MSELoss()
    #loss_fn = MSEplusOpLoss(matrix)


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        name= "scaleinf_dense_trans_poisson_100by100",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-3,
        "layers": 2,
        "optim": "Adam",
        "architecture": "dense",
        "dataset": "smoothpoisson_100by100",
        "epochs": 15,
        "activation": "ELU",
        "loss": "MSEplusOpLoss"
        }
    )


    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, dense, loss_fn, optimizer)
        test(validation_loader, dense, loss_fn)
        
    print("Done!")

    torch.save(dense.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/poisson_trans.pth")
    print("Saved PyTorch Model State to poisson_trans.pth")

    wandb.finish()

    
    
    
    # train U-Net
    UNet.load_state_dict(torch.load("/Users/GUSTO/environments/firedrake/src/gusto/learning/unet_poisson.pth"))
    optimizer = torch.optim.Adam(UNet.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ML_acc_solvers",
        name= "UNet-ResNet-scaleinf_trans-poisson_100by100",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 5e-3,
        "layers": 19,
        "optim": "Adam",
        "architecture": "Res-UNet",
        "dataset": "smoothpoisson_100by100",
        "epochs": 30,
        }
    )
    
    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
    
        train(training_loader, UNet, loss_fn, optimizer)
        test(validation_loader, UNet, loss_fn)
        
    print("Done!")

    torch.save(UNet.state_dict(), "/Users/GUSTO/environments/firedrake/src/gusto/learning/unet_poisson_trans.pth")
    print("Saved PyTorch Model State to unet_poisson_trans.pth")

    wandb.finish()



#avoid running the rest of the script when just importing ML model from elsewhere
if __name__ == '__main__':
    datadir = '/Users/GUSTO/data/training/'
    in_file = datadir + '/vs.csv'
    out_file = datadir + '/ps.csv'
    poissontrain()
    #helmholtztrain()
    transferlearn(in_file, out_file)
    