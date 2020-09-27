# Deep Learning with PyTorch: A 60 Minute Blitz


```python
import torch
import numpy as np
```

## What is PyTorch

### Getting Started

#### Tensors

Construct a 5x3 matrix, unitialized.


```python
x = torch.empty(5, 3)
print(x)
```

    tensor([[9.5461e-01, 4.4377e+27, 1.7975e+19],
            [4.6894e+27, 7.9463e+08, 3.2604e-12],
            [2.6209e+20, 4.1641e+12, 1.9434e-19],
            [3.0881e+29, 6.3828e+28, 1.4603e-19],
            [7.7179e+28, 7.7591e+26, 3.0357e+32]])


A randomly initialized matrix.


```python
x = torch.rand(5, 3)
print(x)
```

    tensor([[0.4489, 0.1336, 0.5693],
            [0.3049, 0.7561, 0.9073],
            [0.1251, 0.9037, 0.3331],
            [0.8124, 0.3350, 0.2602],
            [0.2203, 0.7381, 0.0535]])


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

    tensor([[-2.0303,  0.4544,  2.0180],
            [-0.6643,  1.5354,  0.4815],
            [ 0.7523,  1.2000, -0.4307],
            [ 0.3862, -1.4694, -0.5800],
            [-0.6713, -0.8095, -1.7725]])


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

    tensor([[-1.6251,  1.3736,  2.5858],
            [-0.0808,  2.4137,  0.7262],
            [ 0.9908,  1.3316, -0.3370],
            [ 0.6108, -1.3220, -0.4926],
            [-0.3491, -0.4727, -1.7639]])



```python
torch.add(x, y)
```




    tensor([[-1.6251,  1.3736,  2.5858],
            [-0.0808,  2.4137,  0.7262],
            [ 0.9908,  1.3316, -0.3370],
            [ 0.6108, -1.3220, -0.4926],
            [-0.3491, -0.4727, -1.7639]])




```python
# provide an output tensor as an argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

    tensor([[-1.6251,  1.3736,  2.5858],
            [-0.0808,  2.4137,  0.7262],
            [ 0.9908,  1.3316, -0.3370],
            [ 0.6108, -1.3220, -0.4926],
            [-0.3491, -0.4727, -1.7639]])



```python
# in place
y.add_(x)
print(y)
```

    tensor([[-1.6251,  1.3736,  2.5858],
            [-0.0808,  2.4137,  0.7262],
            [ 0.9908,  1.3316, -0.3370],
            [ 0.6108, -1.3220, -0.4926],
            [-0.3491, -0.4727, -1.7639]])


NumPy-like indexing.


```python
print(x[:, 1])
```

    tensor([ 0.4544,  1.5354,  1.2000, -1.4694, -0.8095])


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

    tensor([-0.0352])
    -0.035218432545661926


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

    <AddBackward0 object at 0x7ff44fa1fbe0>


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


```python

```
