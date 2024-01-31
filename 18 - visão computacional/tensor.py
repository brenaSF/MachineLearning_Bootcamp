import torch
# How to create a tensor object
# 1-D array
sample = torch.tensor([10,11])
sample.shape
# As soon as the bracket ends, bring the numbers to the bottom
x = torch.tensor([[10,11],[1,2]])
x.shape
## [10,11
## 1,2]
y = torch.tensor([[10],[11]])
y.shape
# [10
# 11]

# The code below would help you visualize a 3-D tensor better, where "a" is a 3 dimensional tensor


import torch
import plotly.graph_objs as go
import plotly.offline as pyo

# Create a 3-dimensional tensor
a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Create a list of all the x, y, and z values in the tensor
x_vals = []
y_vals = []
z_vals = []
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        for k in range(a.shape[2]):
            x_vals.append(i)
            y_vals.append(j)
            z_vals.append(k)

# Create a Plotly 3D scatter plot
trace = go.Scatter3d(
    x=x_vals,
    y=y_vals,
    z=z_vals,
    mode='markers',
    marker=dict(
        size=5,
        color=a.flatten(),
        colorscale='Viridis',
        opacity=0.8
    )
)

# Create the plot layout
layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis=dict(title='Dimension 1'),
        yaxis=dict(title='Dimension 2'),
        zaxis=dict(title='Dimension 3')
    )
)

# Combine the trace and layout into a figure and display it
fig = go.Figure(data=[trace], layout=layout)
pyo.plot(fig, filename='tensor.html')






# Tensor with Random Numbers
# torch.zeros((shape tuple))
torch.zeros((3, 4))
torch.zeros((3, 4))
torch.randint(low=0, high=10, size=(3,4))
torch.randint(low=0, high=10, size=(3,4,2))
torch.randint(low=0, high=10, size=(3,4,4))
# From above tensor we can see that size is in the form of (Number of stacks (channels in images, will be covered in the future lessons), rows, columns)
# Think of a cuboid while trying to visualize this tensor

