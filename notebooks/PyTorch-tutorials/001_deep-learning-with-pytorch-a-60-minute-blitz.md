# [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)


```python
import torch
import numpy as np
```

## [What is PyTorch](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)

### Getting Started

#### Tensors

Construct a 5x3 matrix, unitialized.


```python
x = torch.empty(5, 3)
print(x)
```

    tensor([[-1.3032e+26,  4.5832e-41, -1.3032e+26],
            [ 4.5832e-41,         nan,  0.0000e+00],
            [ 7.6194e+31,  1.5564e+28,  4.7984e+30],
            [ 6.2121e+22,  1.8370e+25,  1.4603e-19],
            [ 6.4069e+02,  2.7489e+20,  1.5444e+25]])


A randomly initialized matrix.


```python
x = torch.rand(5, 3)
print(x)
```

    tensor([[0.3105, 0.5768, 0.4052],
            [0.7626, 0.5865, 0.5171],
            [0.3287, 0.7484, 0.2569],
            [0.8918, 0.7016, 0.8462],
            [0.3458, 0.8036, 0.2743]])


Construct a tensor from data.


```python
x = torch.tensor([5.5, 3])
print(x)
```

    tensor([5.5000, 3.0000])


Create a tensor based on an existing tensor.
New properties can be supplied when copying.


```python
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)



```python
x = torch.randn_like(x, dtype=torch.float)
print(x)
```

    tensor([[-0.1035,  1.1275, -0.2697],
            [-0.4714,  0.2587, -1.4718],
            [ 0.6178,  1.1679,  0.9228],
            [-0.6413, -0.8245,  0.8542],
            [ 0.5625, -1.1115,  0.2299]])


Can get size of a tensor.
It is returned as a tuple.


```python
x.size()
```




    torch.Size([5, 3])



#### Operations

There are multiple syntaxes for operations.
For example, addition:


```python
y = torch.rand(5, 3)
print(x + y)
```

    tensor([[-0.0847,  1.3182,  0.4889],
            [-0.2606,  0.7407, -1.2082],
            [ 1.5850,  1.2210,  1.8451],
            [-0.4966, -0.5162,  1.2255],
            [ 0.7382, -1.0366,  0.8779]])



```python
torch.add(x, y)
```




    tensor([[-0.0847,  1.3182,  0.4889],
            [-0.2606,  0.7407, -1.2082],
            [ 1.5850,  1.2210,  1.8451],
            [-0.4966, -0.5162,  1.2255],
            [ 0.7382, -1.0366,  0.8779]])




```python
# provide an output tensor as an argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

    tensor([[-0.0847,  1.3182,  0.4889],
            [-0.2606,  0.7407, -1.2082],
            [ 1.5850,  1.2210,  1.8451],
            [-0.4966, -0.5162,  1.2255],
            [ 0.7382, -1.0366,  0.8779]])



```python
# in place
y.add_(x)
print(y)
```

    tensor([[-0.0847,  1.3182,  0.4889],
            [-0.2606,  0.7407, -1.2082],
            [ 1.5850,  1.2210,  1.8451],
            [-0.4966, -0.5162,  1.2255],
            [ 0.7382, -1.0366,  0.8779]])


NumPy-like indexing.


```python
print(x[:, 1])
```

    tensor([ 1.1275,  0.2587,  1.1679, -0.8245, -1.1115])


Resizing and reshaping using `torch.view()`.


```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # -1 is inferred from the other dimensions
print(x.size(), y.size(), z.size())
```

    torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])


Use `.item()` on a one-element tensor to get the Python number.


```python
x = torch.randn(1)
print(x)
print(x.item())
```

    tensor([-0.3706])
    -0.3705705404281616


### NumPy Bridge

Converting a Torch Tensor to a NumPy array and back is easy.
If the Torch Tensor is on CPU, they point to the same part of memory and changes to one will change the other.

#### Converting a Torch Tensor to a NumPy Array 


```python
a = torch.ones(5)
print(a)
```

    tensor([1., 1., 1., 1., 1.])



```python
b = a.numpy()
print(b)
```

    [1. 1. 1. 1. 1.]



```python
a.add_(1)  # add 1 in-place
print(a)
print(b)
```

    tensor([2., 2., 2., 2., 2.])
    [2. 2. 2. 2. 2.]


#### Converting NumPy Array to Torch Tensor


```python
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

    [2. 2. 2. 2. 2.]
    tensor([2., 2., 2., 2., 2.], dtype=torch.float64)


### CUDA Tensors

Tensors can be moved onto any device using the `.to()` method.


```python
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)

    x = x.to(device)

    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
else:
    print("CUDA is not available.")
```

    CUDA is not available.


---

## [Autograd: Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#)

The `autograd` package provides the automatic differentiation that is central to all PyTorch neural networks.
It is "define-by-run" framework meaning that the backpropogation is defined by how the code is run.
This means that each iteration can be different.

### Tensor

The `torch.Tensor` is the central class of the package.
If `.requires_grad` is set to `True`, all operations on it will be tracked.
Once the computation is finished, `.backward()` can be called to have the gradient computed automatically.
The gradient is accumulated into the `.grad` attribute.

`.detach()` can be called on a tensor to stop tracking during a computation.

To prevent tracking history (and using memory), the code can be wrapped `with torch.no_grad():`.
This can be helpful when eveluating a model that has trainable parameters that require `requires_grad=True`, but the gradients are actually needed.

The `Function` class is also very important for the `autograd` package.
`Tensor` and `Function` build an acyclic graph with a complete hisotry of the computation.
Each tensor has a `.grad_fn` attribute that references a `Function` that created the `Tensor`

A call to `.backward()` will compute the derivatives of a `Tensor`.
Unless the tensor is a scalar (holds a single element of data), another tensor must be supplied to the `gradient` argument.

Create a tesnor and set `.requires_grad=True` to track computation.


```python
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)


Perform a tensor operation.


```python
y = x + 2
print(y)
```

    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)


`y` was created as a result of an operation, so it has a `grad_fn` attribute.


```python
print(y.grad_fn)
```

    <AddBackward0 object at 0x7fc35ecbbdc0>


Perform more operations on `y`.


```python
z = y * y * 3
out = z.mean()

print(z, out)
```

    tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)


### Gradients

Now we can perform backpropagation.

Since `out` is a single scalar, we can call `.backward()` without passing any parameters.


```python
out.backward()
```

The gradient $\frac{d(out)}{dx}$ can now be printed.


```python
print(x.grad)
```

    tensor([[4.5000, 4.5000],
            [4.5000, 4.5000]])


---
## [Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

Neural networks (NN) are constructed using the `torch.nn` package.
It uses `autograd` to define and differentiate them.
An `nn.Module` contains layers and a method `.forward(input)` that returns the output.

A typical training procedure:

1. define the NN with some learnable parameters (weights)
2. iterate over the dataset of inputs
3. process the inputs through the network
4. compute the loss
5. propagate gradients back into the network's parameters
6. update the weights of the network

A typical simple updating rule for the final step is `weight = weight - learning_rate * gradient`.

### Define the network

Below is a example neural network.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input channel, 6 output channels, 3x3 square convolution
        
        #kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6x6 from image dimensions
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # Max pooling over a (2x2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```


```python
net = Net()
print(net)
```

    Net(
      (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
      (fc1): Linear(in_features=576, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )


The `forward()` method must be defined and the `backward()` method is automatically defined using `autograd`.

The learnable parameters of a model are returned by `net.parameters()`.


```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # the weights of conv1
```

    10
    torch.Size([6, 1, 3, 3])


We can try feeding in a random 32x32 input.


```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

    tensor([[-0.0503,  0.0191,  0.0587,  0.0462, -0.0709, -0.1065,  0.0905, -0.0233,
              0.1208, -0.0207]], grad_fn=<AddmmBackward>)


Zero the gradient buffers of all parameters and backprops with random gradients.


```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```

Note that `torch.nn` only supports mini-batches of input, not single samples.
`input.unsqueeze(0)` can be used to fake batch dimensions.

### Loss Function

A loss function takes the model's output and target as inputs and computes a value that estimates how far away the outputs are from the target.

There are several available in the `nn` package.
A simple one is `nn.MSELoss` which computes the mean-squared error between the input and target.
Below is an example.


```python
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)  # mimic the shape of the output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

    tensor(1.6354, grad_fn=<MseLossBackward>)


Now, when we call `loss.backward()`, the whole graph is differentiated w.r.t. the loss and all Tensors in the graph (that has `requires_grad=True`) will have their `.grad` Tensor accumulated with the gradient

### Backprop

To backprogagate the error, all that is needed is to call `loss.backward()`.
Make sure to clear the existing gradients, first, else gradients will be accumulated to existing gradients.

Below, we call `loss.backward()` and look at `conv1`'s bias gradients before and after the backprop.


```python
net.zero_grad()

print("conv1.bias.grad before backward.")
print(net.conv1.bias.grad)

loss.backward()

print("conv1.bias.grad after backward.")
print(net.conv1.bias.grad)
```

    conv1.bias.grad before backward.
    tensor([0., 0., 0., 0., 0., 0.])
    conv1.bias.grad after backward.
    tensor([-0.0052, -0.0034,  0.0013, -0.0077,  0.0130,  0.0208])


### Update the weights

The simplest rule used in practice is the Stochastic Gradient Descent (SGD):

```
weight = weight - learning_rate * gradient
```

Various updating rules have been updatig in `torch.optim`.
An example of using it is shown below.


```python
import torch.optim as optim

# Create the optimizer.
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()  # zero the gradient buffers

output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # performs the update
```


```python
print(net.conv1.bias.grad)
```

    tensor([-0.0038,  0.0008,  0.0008, -0.0175,  0.0033,  0.0206])


## [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)

### What about data?

Generally, the data can be loaded as NumPy arrays using standard libraries and then converted into `torch.*Tensor` objects.
Here are some standard libraries for these operations:

- images: Pillow, OpenCV
- audio: scipy, librosa
- text: standard Python, NLTK, SpaCy

The `torchvision` package was created specifically for vision tasks.
It has data loaders for common data sets (e.g. Imagenet, CIFAR10, MNIST) and data transformers for images, vizualization: `torchvision.datasets` and `torch.utils.data.DataLoader`.

For the following tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

### Training and image classifier


```python

```
