import torch
# Multiplication
x = torch.tensor([[1,2,3,4], [5,6,7,8]]) 
print(x.shape)
x*10
# Sum
x.add(10)
# Reshape
## There are multiple ways to reshape a tensor
## 1) view
x
x.shape
x.view(4,2)
## While Reshaping, we need to keep in mind that the number of elements inside the tensor should remain the same, like in the above example we had 8 elements in total, the possible shapes to which it can be converted should have 8 elements
x.view(8,1)

## 2) Squeeze
## In squeeze, we provide the axis index that we want to remove which has only 1 item in that dimension
a = torch.ones(2,1,10)
# This is a tensor of 2 "stacks" having one row and 10 columns each, total 20 elements
a
print(f"This is the shape before squeezing {a.shape}")
squeezed = a.squeeze(1)
print(f"This is the shape after squeezing {squeezed.shape}")
## As you can see we removed the first axis index which only had 1 item. If the axis has more than one item, it won't be removed
a = torch.ones(2,2,10)
print(f"This is the shape before squeezing {a.shape}")
squeezed = a.squeeze(1)
print(f"This is the shape after squeezing {squeezed.shape}")
a = torch.ones(2,3,1)
print(f"This is the shape before squeezing {a.shape}")
squeezed = a.squeeze(2)
print(f"This is the shape after squeezing {squeezed.shape}")

## 3) Unsqueeze
## It is the exact opposite of squeeze, where instead of removing, we are adding a dimension to our tensor
a = torch.ones(2,3)
print(f"This is the shape before squeezing {a.shape}")
squeezed = a.unsqueeze(2)
print(f"This is the shape after squeezing {squeezed.shape}")

## [None] indexing: A very common approach
a = torch.ones(2,3)
# Adding none would auto create a fae dimension at the specified axis
print(a[None].shape) # Fake axis at 0 index
print(a[:,None].shape) # Fake axis at 1 index
print(a[:,:,None].shape) # Fake axis at 2 index

# In computer vision, keep in mind that the shape of tensor would be:
## (total number of elements) for 1-D tensor
## (rows,columns) for 2-D tensor
## (number of channels, rows, columns) for 3-D tensor
## (batch size, num of channels, rows, columns) for 4-D tensor
## In future lectures you would see how rows and columns turn into height and width of the image giving us (batch size, num channels, height, width) tensors



import torch
# Matrix Multiplication
## If A and B are two matrices, we can multiply them together only if they are of shape A (nxm) B(mxz) which would give us a matrix of shape C(nxz)
x = torch.tensor([[1,2,3,4],[5,6,7,8]]) # shape = (2,4)
y = torch.tensor([[1,2,3],
                  [2,3,4],
                  [4,5,6],
                  [7,8,9]]) # shape = (4,3)
# using torch.matmul()
torch.matmul(x, y) # shape = (2,3)

# Concatenation
x = torch.randn(1,4,5)
z = torch.cat([x,x], axis=1)
print('Concatenated axis 1:', x.shape, z.shape)
# z tensor has shape (1,8,5)

## Permute
x = torch.randn(3,20,10)
z = x.permute(2,0,1)
print('Permute dimensions:', x.shape, z.shape)
## Think of the above tensor as a coloured image (having 3 channels) and a height and width of (20,10). Axis 0: Channels, Axis 1: Height, Axis 2: Width. When we permute (let's say to (2,0,1)), we swap the axis in the order defined by the permute operation. So now our shape becomes (10,3,20)


















