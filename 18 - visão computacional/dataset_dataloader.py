# Batch Size: Batch size refers to the number of data points considered to calculate the loss value or update weights
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
# Define the same dataset as in the previous lessons
x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]
X = torch.tensor(x).float()
Y = torch.tensor(y).float()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)
#In the MyDataset class, we hold the necessary details for retrieving individual data points, allowing us to group them into batches (using DataLoader) and process #them together in a single forward and back-propagation step to update the weights.
# Theser are 3 main things you need to remember
## 1) Inherit from Dataset class and implement __init__ method
## 2) Implement __getitem__ method (Whatever this method returns is what we get when we create a dataloader)
## 3) Implement __len__ method
## These 3 functions are a necessity, there is also a collate_fn which I would cover in the future lessons
class MyDataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x).float().to(device)
    self.y = torch.tensor(y).float().to(device)
  def __len__(self):
    return len(self.x)
  def __getitem__(self,ix):
    return self.x[ix], self.y[ix]
ds = MyDataset(x,y)
# Dataloader object is used to load data from a dataset and return it in the form of mini-batches. It provides an iterable over the dataset, with support for multi-process data loading and customizable data transformation.
# Notice Batch size
dl = DataLoader(ds, batch_size=2, shuffle=True)
# To load the data we loop through it
for x,y in dl:
  print(x,y)

# Using the DataLoader object in the training code
import torch
import torch.nn as nn
from torch.optim import SGD

class MyNeuralNet(nn.Module):
  def __init__(self):  
    # When we call the super.__init__() method we ensure we are inhertiting   
    super().__init__()
    self.layer1 = nn.Linear(2,8) # A linear layer
    self.activation = nn.ReLU() # activation function
    self.layer2 =  nn.Linear(8,1)

  # When we pass something through the model object, it calls the forward function 
  def forward(self,x):
    x = self.layer1(x)
    x = self.activation(x)
    x = self.layer2(x)
    return x
model = MyNeuralNet()
loss_func = nn.MSELoss()
opt = SGD(model.parameters(), lr = 0.001)
losses = []
for _ in range(50): #Running for 50 epochs
  for data in dl:
    opt.zero_grad() # Setting gradients to zero before every epoch
    x1, y1 = data
    loss_value = loss_func(model(x1),y1)
    #  the gradients of the loss function with respect to all the trainable parameters of the network are computed and stored in the grad attribute of the corresponding tensors.
    loss_value.backward()

    # opt.step() is to update the weights and biases of the neural network using the computed gradients and the chosen optimization algorithm
    opt.step() 
    losses.append(loss_value.detach().numpy())
# By using a dataloader object we are able to train the model much faster, as batch processing is taking place
















