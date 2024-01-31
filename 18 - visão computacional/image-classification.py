# Working with FMNIST Dataset
from torchvision import datasets

import torch
data_folder = '~/data/FMNIST' # This can be any directory # you want to download FMNIST to
fmnist = datasets.FashionMNIST(data_folder, download=True,train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets
import matplotlib.pyplot as plt

import numpy as np
R, C = len(tr_targets.unique()), 10
fig, ax = plt.subplots(R, C, figsize=(10,10))
for label_class, plot_row in enumerate(ax):    
  label_x_rows = np.where(tr_targets == label_class)[0]
  for plot_cell in plot_row:
    plot_cell.grid(False); plot_cell.axis('off')
    ix = np.random.choice(label_x_rows)
    x, y = tr_images[ix], tr_targets[ix]
    plot_cell.imshow(x, cmap='gray')
plt.tight_layout()
# Each row represents sample of images belonging to the same class


# Converting Images to Tensor (Creating the dataset)
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchvision import datasets
class FMNISTDataset(Dataset):
  def __init__(self,x,y):
    x = x.float()
    # We are flattening each image, h=w=28
    # -1 means other dimennsion would adjust automatically, based on the number elements
    x = x.view(-1,28*28)
    self.x, self.y = x,y
  def __getitem__(self,idx):
    x,y = self.x[idx], self.y[idx]
    return x.to(device), y.to(device)
  def __len__(self):
    return len(self.x)
def get_data():     
  train = FMNISTDataset(tr_images, tr_targets)     
  trn_dl = DataLoader(train, batch_size=32, shuffle=True)    
  return trn_dl


# Training the model
from torch.optim import SGD
def get_model():    
  model = nn.Sequential(                
      nn.Linear(28 * 28, 1000),                
      nn.ReLU(),                
      nn.Linear(1000, 10)            
      ).to(device)    
  loss_fn = nn.CrossEntropyLoss()    
  optimizer = SGD(model.parameters(), lr=1e-2)    
  return model, loss_fn, optimizer
@torch.no_grad()
def accuracy(x, y, model):    
  model.eval() 
  prediction = model(x)
  max_values, argmaxes = prediction.max(-1)    
  is_correct = argmaxes == y    
  return is_correct.cpu().numpy().tolist()
def train_batch(x, y, model, opt, loss_fn):
    model.train()    
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    # Flush gradients memory for next batch of calculations
    optimizer.zero_grad()
    return batch_loss.item()
trn_dl = get_data()
model, loss_fn, optimizer = get_model()
losses, accuracies = [], []
for epoch in range(5):
    print(epoch)
    epoch_losses, epoch_accuracies = [], []
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        epoch_losses.append(batch_loss)
    epoch_loss = np.array(epoch_losses).mean()
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        epoch_accuracies.extend(is_correct)
    epoch_accuracy = np.mean(epoch_accuracies)
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)
epochs = np.arange(5)+1

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.legend()








# Scaling the Dataset
# Since the pixel values belong from 0-255, we can scale our dataset simply by diving the pixel values with 255

class FMNISTDataset(Dataset):
  def __init__(self,x,y):
    x = x.float()/255.
    x = x.view(-1,28*28)
    self.x, self.y = x,y
  def __getitem__(self,idx):
    x,y = self.x[idx], self.y[idx]
    return x.to(device), y.to(device)
  def __len__(self):
    return len(self.x)

















