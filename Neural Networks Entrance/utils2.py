#!/usr/bin/env python
# coding: utf-8

# # Towards Multi-Layer Neural Networks

# Here, you will build and run multi-layer (and deep) neural networks! We've broken this assignment up into two parts.
# - `HW2-Part1` will walk you through implementing all the fuctions to build a deep network
# -  You will save all the methods (functions) that you worked on this notebook into a single .py file and you will save it as a `utils2.py`. You will use those functions (as a part of the `utils2.py` script) in Part2.
# - `HW2-Part2` will compose these functions into a deep network for image classification
# - In this file, in each method, there is a code snipped is given to you and some parts of that code is missing. You are required to complete those missing parts (mostly highlighted with "None", so that the code can run and give its expected output.
# 
# <div class="alert alert-warning" markdown="1">
#     <strong>After this assignment, you should be able to:</strong>
#     <ul> 
#         <li>Use non-linear units using activations functions such as ReLU to improve your model</li>
#         <li>Build multi-layer (deep) neural networks </li>
#     </ul>
# </div>

# **A quick note on notation:**
# - Superscript $[l]$ tells you a quantity associated with the $l^{th}$ layer
#   - e.g. $a^{[l]}$ is the $l^{th}$ layer's activation; $W^{[l]}$ and $b^{[l]}$ are the parameters: weight and bias for the $l^{th}$ layer.
# - Supercript $(i)$ is a quantity associated with the $i^{th}$ sample
#   - e.g. $x^{(i)}$ is the $i^{th}$ training sample
# - Subscript $i$ means the $i^{th}$ entry of a vector (of a sample)
#   - e.g. $a^{[l]}_i$ is the $i^{th}$ entry of the $l^{th}$ layer's activation

# ## 1 - Packages
# 
# Let's start with importing all the packages we'll use in this assignment.
# 
# - [`numpy`](www.numpy.org) is a package for scientific computing with Python.
# - [`matplotlib`](http://matplotlib.org) is for plotting graphs in Python.
# - `np.random.seed(1)` is used for getting the same results after using random function calls. **This will help us grade your work – changing this will likely result in a failing grade.** _Don't remove or edit the lines running np.random.seed function._

# In[1]:


import numpy as np
import h5py

np.random.seed(1)


# ## 2 - What we'll do throughout the assignment
# 
# You will first implement multiple "helper functions." These functions will also be used in `Part2` to build `2-layer` and `L-layer` neural networks. We'll walk you through each helper function you'll implement and have fairly detailed instructions on the necessary steps.
# 
# An outline of what you will do in this assignment:
# - Initialize parameters for a `2-layer` and an `L-layer` neural network
# - Implement forward propagation (purple, in the figure below)
#   - Complete forward propagation's `linear` combination (this results in $Z^{[L]}$)
#   - (We'll give you the `activation` function – `ReLU`/`Sigmoid`)
#   - Combine the `linear` and `activation` steps into a single \[`linear` -> `activation`\] step
#   - Stack the function from the previous step into $L-1$ layers of \[`linear` -> `ReLU`\] and add a final \[`linear` -> `Sigmoid`\] layer at the end; this will result in `L_model_forward`
# - Compute the cost
# - Implement backpropagation (red, in the figure below)
#   - Complete backpropagation's `linear` combination (this results in the function: $Z^{[L]}$)
#   - (We'll give you the gradient of the `activation` function  – `ReLU_backward` / `Sigmoid_backward`)
#   - Combine the `linear` and `activation` steps into a single \[`linear` -> `activation`\] step
#   - Stack the function from the previous step into $L-1$ copies of \[`linear` -> `ReLU`\] and add \[`linear` -> `Sigmoid`\] for the final layer; this will result in the function: `L_model_backward`
# - Lastly, update the parameters (weights and bias values).
# 
# <img src="images/final outline.png" style="width:800px;height:500px;">
# <caption><center> **Figure 1**</center></caption><br>
# 
# 
# <div class="alert alert-warning">
# <strong>NOTE:</strong> Every forward function has a corresponding backward function. That's why, at every step of your forward propagation, you'll be storing some values in a cache. These cached values are needed to compute the gradients &ndash; in the backpropagation module, you'll then use the cache to calculate the gradients. We'll walk you through how to carry out each of these steps.
# </div>

# ## 3 - Initialization
# 
# Below, you will implement two helper functions to initialize the parameters of your model.
# 1. First one will be used to `initialize_parameters`
# 2. Then, the second one will generalize `initialize_parameters_deep` to $L$ layers
# 
# ### 3.1 - Initialization for a `2-layer` Neural Network
# 
# <div class="alert alert-info"> <h2>Exercise 1:</h2>
#     <p> define and initialize the parameters of a <code>2-layer</code> neural network. </p>
#     <ul><strong>Instructions:</strong>
#         <li> The model structure should be: <code>input -> linear -> relu -> linear -> sigmoid</code>.</li>
#         <li> Random initialization should be used for the weight matrices. (Hint: Use <code>np.random.randn(shape) * 0.01</code>.)</li>
#         <li> For the biases, use a zero initialization (Hint: use <code>np.zeros(shape)</code>).</li>
#     </ul>
# </div>

# In[2]:


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    ### START CODE HERE ### (~4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1)) 
    ### END CODE HERE ###
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1, "W2": W2,
                  "b1": b1, "b2": b2}
    
    return parameters  


# In[3]:


# test your function below and compare your result to the expected output

parameters = initialize_parameters(3,3,1)
print("W1 values = " + str(parameters["W1"]))
print("b1 values = " + str(parameters["b1"]))
print("W2 values = " + str(parameters["W2"]))
print("b2 values = " + str(parameters["b2"]))


# **Expected output:**
# 
# ```python
# W1 values = [[ 0.01624345 -0.00611756 -0.00528172]
#  [-0.01072969  0.00865408 -0.02301539]
#  [ 0.01744812 -0.00761207  0.00319039]]
# b1 values = [[0.]
#  [0.]
#  [0.]]
# W2 values = [[-0.0024937   0.01462108 -0.02060141]]
# b2 values = [[0.]]
# ```

# ### 3.2 - Initialization of an `L-layer` Neural Network
# 
# Initializating an `L-layer` neural network is more complex, largely because of the increased number of weight matrices and bias vectors. While completing the function: `initialize_parameters_deep` below, make sure that the dimensions between layers match! This is a common point of error, especially while learning. (Remember that, for example, $n^{[1]}$ means the number of units in Layer $1$.) If the size of the input $X$ is $(12288, 209)$ with $m=209$ examples, then...
# 
# | Layer | `W.shape`                | `b.shape`        | Activation                                    |`activation.shape`|
# |:------:|--------------------------|------------------|-----------------------------------------------|------------------|
# |**1**   | $(n^{[1]}, 12288)$       | $(n^{[1]}, 1)$   | $Z^{[1]} = W^{[1]}  X + b^{[1]}$              | $(n^{[1]}, 209)$ |
# |**2**   | $(n^{[2]}, n^{[1]})$     | $(n^{[2]}, 1)$   | $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$         | $(n^{[2]}, 209)$ |
# |$\vdots$| $\vdots$                 | $\vdots$         | $\vdots$                                      | $\vdots$         |
# |**L-1** | $(n^{[L-1]}, n^{[L-2]})$ | $(n^{[L-1]}, 1)$ | $Z^{[L-1]} = W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ | $(n^{[L-1]}, 209)$ |
# |**L**   | $(n^{[L]}, n^{[L-1]})$   | $(n^{[L]}, 1)$   | $Z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}$       | $(n^{[L]}, 209)$ |
# 
# Recall, from PA1, that computing $W X + b$ in `numpy` will apply broadcasting; e.g. if:
# 
# $$ W = \begin{bmatrix}
#     j  & k  & l\\
#     m  & n & o \\
#     p  & q & r 
# \end{bmatrix}\;\;\; X = \begin{bmatrix}
#     a  & b  & c\\
#     d  & e & f \\
#     g  & h & i 
# \end{bmatrix} \;\;\; b =\begin{bmatrix}
#     s  \\
#     t  \\
#     u
# \end{bmatrix}\tag{2}$$
# 
# Then $WX + b$ will be:
# 
# $$ WX + b = \begin{bmatrix}
#     (ja + kd + lg) + s  & (jb + ke + lh) + s  & (jc + kf + li)+ s\\
#     (ma + nd + og) + t & (mb + ne + oh) + t & (mc + nf + oi) + t\\
#     (pa + qd + rg) + u & (pb + qe + rh) + u & (pc + qf + ri)+ u
# \end{bmatrix}\tag{3}  $$

# <div class="alert alert-info"> <h2>Exercise 2:</h2>
#     <p> Implement a function to initialize the parameters for an <code>L-layer</code> neural network. </p>
#     <ul><strong>Instructions:</strong>
#         <li> The model's structure is <code>([linear -> relu] * (L-1)) ->linear -> sigmoid</code>. i.e. the model has $L-1$ layers using a <code>ReLU</code> activation, followed by an output layer with a <code>Sigmoid</code> activation.</li>
#         <li> Random initialization needs to be used for the weights. (Hint: Use <code>np.random.randn(shape) * 0.01</code>.)</li>
#         <li> For the biases, use a zero initialization (Hint: Use <code>np.zeros(shape)</code>).</li>
#         <li> We'll store $n^{[l]}$, the number of units in different layers, in <code>layer_dims</code>. </li>
#     </ul>
# </div> 
# 
# For example, consider a `3-layer` neural network where `layer_dims = [2, 4, 1]`. This corresponds to an input layer with 2 units, one hidden layer with 4 units, and 1 output layer with a single unit. This means that `W1.shape = (4, 2)`, `b1.shape = (4, 1)`, `W2.shape = (1, 4)`, and `b2.shape = (1,1)`; now, we'll generalize this to $L$ layers! 
# 
# The code below is an implementation for $L=1$ (a single-layer neural network); this should inspire you to implement the general case `L-layer` neural network.
# ```python
#     if L == 1:
#         parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
#         parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))
# ```

# In[24]:


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


# In[25]:


# test your function below and compare your result to the expected output

parameters = initialize_parameters_deep([5,5,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# **Expected output**:
#        
# ```python
# W1 = [[ 0.01788628  0.0043651   0.00096497 -0.01863493 -0.00277388]
#  [-0.00354759 -0.00082741 -0.00627001 -0.00043818 -0.00477218]
#  [-0.01313865  0.00884622  0.00881318  0.01709573  0.00050034]
#  [-0.00404677 -0.0054536  -0.01546477  0.00982367 -0.01101068]
#  [-0.01185047 -0.0020565   0.01486148  0.00236716 -0.01023785]]
# b1 = [[0.]
#  [0.]
#  [0.]
#  [0.]
#  [0.]]
# W2 = [[-0.00712993  0.00625245 -0.00160513 -0.00768836 -0.00230031]
#  [ 0.00745056  0.01976111 -0.01244123 -0.00626417 -0.00803766]
#  [-0.02419083 -0.00923792 -0.01023876  0.01123978 -0.00131914]]
# b2 = [[0.]
#  [0.]
#  [0.]]
# ```

# ## 4 - Implementing Forward Propagation 
# 
# ### 4.1 - `linear` Forward 
# Now that we've initialized the parameters, we can compute the forward propagation. For that, you'll start by implementing some basic functions that we'll use later on when building the model. Complete the implementation of the followed steps, in order:
# 
# 1. `linear`
# 2. `linear -> activation`, where `activation` can be either ReLU or Sigmoid
# 3. `([linear -> ReLU]` $\times$ `(L - 1)) -> linear -> sigmoid` (this is the whole model)
# 
# The forward module (vectorized over all examples) computes the following equations:
# $$Z^{[L]} = W^{[L]}A^{[L-1]} + b^{[L]}\tag{4}$$ where $A^{[0]} = X$
# 
# <div class="alert alert-info"><h2> Exercise 3:</h2>
#     <p>Build the linear part of a layer for forward propagation.</p>
#     <p>
#         <strong>Reminder:</strong>
#         The mathematical expression of that for layer $l$ is $Z^{[L]} = W^{[L]}A^{[L-1]} + b^{[L]}$. Hint: <code>np.dot</code> might be a useful here. Also, be sure to use (print) <code>W.shape</code> to verify dimensions, if you get errors related to the dims!
#     </p>
# </div>

# In[26]:


# GRADED FUNCTION: linear_forward

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    ### START CODE HERE ### (~1 line of code)
    Z = W.dot(A) + b
    ### END CODE HERE ###
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


# In[27]:


# test your function below and compare your result to the expected output

np.random.seed(1)
A = np.random.randn(3,2)
W = np.random.randn(1,3)
b = np.random.randn(1,1)

Z, linear_cache = linear_forward(A, W, b)
print("linear part: Z = " + str(Z))


# **Expected output**:
# ```python
# linear part: Z = [[ 3.26295337 -1.23429987]]
# ```

# ### 4.2 -  `linear -> activation` Forward
# 
# In this assignment, we'll be using two activation functions:
# 
# 1. **Sigmoid**: $\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$. We've provided the `sigmoid` function for you. This function returns **two** items: the activation value `A` and a `cache` which contains `Z` (the `cache` is what we'll feed into the corresponding backward function). Use it like so:
# ``` python
# A, activation_cache = sigmoid(Z)
# ```
# 
# 1. **ReLU**: The mathematical representation is $A = ReLU(Z) = max(0, Z)$. We've provided the `relu` function for you. This function returns **two** items: the activation value `A` and a `cache` which contains `Z` (the `cache` is what we'll feed into the corresponding backward function). Use it like so:
# ``` python
# A, activation_cache = relu(Z)
# ```

# For convenience, you'll group two functions (`linear` and `activation`) into a single function (`linear_activation`). This function will, then, do the `linear` forward step followed by an `activation` forward step.
# 
# <div class="alert alert-info"><h2>Exercise 4</h2>
#     <p> Implement the forward propagation of the <code>linear -> activation</code> step. The mathematical formulation: $A^{[l]} = g(Z^{[l]}) = g(W^{[l]}A^{[l-1]} + b^{[l]})$ where the activation <code>g</code> can be <code>sigmoid</code> or <code>relu</code>. (Use the function <code>linear_forward</code> that you just implemented in the previous step and then choose the appropriate activation function.)</p>
# </div>

# In[28]:


# GRADED FUNCTIONS: Sigmoid, relu, linear_activation_forward

def sigmoid(Z):
    """
    Implement the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    ### START CODE HERE ### (~1 lines of code)
    A = 1 / (1 + np.exp(-Z))
    ### END CODE HERE ###
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z (you can use np.maximum() function.)
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    ### START CODE HERE ### (~1 lines of code)
    A = np.maximum(0, Z)
    ### END CODE HERE ###
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache




# GRADED FUNCTION: linear_activation_forward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        
        ### START CODE HERE ### (~2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        ### END CODE HERE ###
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        
        ### START CODE HERE ### (~2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        ### END CODE HERE ###
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# In[29]:


# test your function below and compare your result to the expected output


np.random.seed(2)
A_prev = np.random.randn(3,2)
W = np.random.randn(1,3)
b = np.random.randn(1,1)

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid : A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU    : A = " + str(A))


# **Expected output**:
# ```python    
# With Sigmoid : A = [[ 0.96890023  0.11013289]]
# With ReLU    : A = [[ 3.43896131  0.        ]]
# ```

# <div class="alert alert-warning"><strong>NOTE:</strong> In deep learning, we count <code>[LINEAR->ACTIVATION]</code> as a single layer, not as two.</div>

# ### d) `L-Layer` Model 
# 
# For further convenience, when implementing an $L$-layer neural network, we'll need a function which replicates `linear_activation_forward` (with ReLU) $L-1$ times, then sets the last layer to `linear_activation_forward` (with Sigmoid).
# 
# <img src="images/model_architecture_kiank.png" style="width:600px;height:300px;">
# <caption><center> **Figure 2** : *[LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID* model</center></caption><br>
# 
# <div class="alert alert-info"><h2>Exercise 5</h2>
#     <p>Implement the forward propagation for the above model.</p>
#     <p><strong>Instructions:</strong> In the code below, <code>AL</code> denotes $A^{[L]} = \sigma(Z^{[L]}) = \sigma(W^{[L]} A^{[L-1]} + b^{[L]})$. (This is also typically called <code>yhat</code>, in papers, you'll likely see that as $\hat{Y}$.)</p>
# </div>
# 
# **Tips**:
# - Use the functions you've previously written
# - `for` loops to replicate `[linear -> relu]`, ($L-1$) times, are a good idea
# - Keep track of the `caches` in the `caches` list! We can add new values (say, `c`) to a `list` just by using `list.append(c)`.

# In[30]:


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        ### START CODE HERE ### (~2 lines of code)
        A, cache = linear_activation_forward(A_prev, parameters['W{:d}'.format(l)], parameters['b{:d}'.format(l)], activation='relu')
        caches.append(cache)
        '''
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
        '''
        ### END CODE HERE ###
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (~2 lines of code)
    AL, cache = linear_activation_forward(A, parameters['W%d' % L], parameters['b%d' % L], activation='sigmoid')
    caches.append(cache)
    '''
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    '''
    
    ### END CODE HERE ###
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


# In[11]:


# test your function below and compare your result to the expected output


def L_model_forward_test_case_2hidden():
    np.random.seed(6)
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,5)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(3,4)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
  
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return X, parameters

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


# **Expected output:**
# ```python
# AL = [[ 0.03921668  0.70498921  0.19734387  0.04728177]]
# Length of caches list = 3
# ```

# <div class="alert alert-success">Great! Now you have a fully-functional implementation of forward propagation that takes the input $X$ and outputs a row vector $A^{[L]}$ containing your predictions. It also records all the intermediate values in <code>caches</code>. Using $A^{[L]}$, we can compute the cost of your predictions.</div>

# ## 5 - Cost function
# 
# Now, we'll implement forward **_and_** backward propagation steps. You'll also compute the cost to see if your model is learning.
# 
# <div class="alert alert-info"><h2>Exercise 6</h2>
#     <p>Compute cross-entropy cost $J$ for logistic regression, using the following formula: $$-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)) \tag{7}$$</p>
# </div>

# In[12]:


# GRADED FUNCTION: compute_cost

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (~1 lines of code)
    cost = -1 / m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))

    ### END CODE HERE ###

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


# In[13]:


# test your function below and compare your result to the expected output


Y = np.asarray([[1, 1, 1]])
aL = np.array([[.8,.9,0.4]])

print("cost = " + str(compute_cost(aL, Y)))


# **Expected Output**:
# ```python
# cost = 0.414931599615397
# ```

# ## 6 - Backward propagation module
# 
# Similar to the forward propagation, you'll also implement helper functions for the backpropagation (backprop). **Remember** backprop is used to calculate the derivatives of the cost function, with respect to the parameters so that we can update the parameters. 
# 
# **Reminder**: 
# <img src="images/backprop_kiank.png" style="width:650px;height:250px;">
# <caption><center> **Figure 3** : Forward and Backward propagation for *LINEAR->RELU->LINEAR->SIGMOID* <br> *The purple blocks represent the forward propagation, and the red blocks represent the backward propagation.*  </center></caption>
# 
# <!-- 
# For those of you who are expert in calculus (you don't need to be to do this assignment), the chain rule of calculus can be used to derive the derivative of the loss $\mathcal{L}$ with respect to $z^{[1]}$ in a 2-layer network as follows:
# 
# $$\frac{d \mathcal{L}(a^{[2]},y)}{{dz^{[1]}}} = \frac{d\mathcal{L}(a^{[2]},y)}{{da^{[2]}}}\frac{{da^{[2]}}}{{dz^{[2]}}}\frac{{dz^{[2]}}}{{da^{[1]}}}\frac{{da^{[1]}}}{{dz^{[1]}}} \tag{8} $$
# 
# In order to calculate the gradient $dW^{[1]} = \frac{\partial L}{\partial W^{[1]}}$, you use the previous chain rule and you do $dW^{[1]} = dz^{[1]} \times \frac{\partial z^{[1]} }{\partial W^{[1]}}$. During the backpropagation, at each step you multiply your current gradient by the gradient corresponding to the specific layer to get the gradient you wanted.
# 
# Equivalently, in order to calculate the gradient $db^{[1]} = \frac{\partial L}{\partial b^{[1]}}$, you use the previous chain rule and you do $db^{[1]} = dz^{[1]} \times \frac{\partial z^{[1]} }{\partial b^{[1]}}$.
# 
# This is why we talk about **backpropagation**.
# !-->
# 
# Similar to forward propagation, we'll be building backprop in three steps:
# 1. `linear` backward
# 1. `linear -> activation` backward, where `activation` computes the derivative of either `ReLU` or `Sigmoid` activations
# 1. `([linear -> relu]` $\times$ `(L - 1)) -> linear -> sigmoid_backward` (this is the complete model for our L-layer neural network)

# ### 6.1 - Linear backward
# 
# For layer $l$, the linear part is: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ (followed by an activation).
# 
# Suppose you've already calculated the derivative $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$. Then, after that, you'll want to calculate the values: $dW^{[l]}, db^{[l]}$ and $dA^{[l-1]}$.
# 
# <img src="images/linearback_kiank.png" style="width:250px;height:300px;">
# <caption><center> **Figure 4** </center></caption>
# 
# Three outputs $(dW^{[l]}, db^{[l]}, dA^{[l]})$ are computed using the input $dZ^{[l]}$. Below, you'll find the formulas you need:
# $$ dW^{[l]} = \frac{\partial \mathcal{L} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} \tag{8}$$
# $$ db^{[l]} = \frac{\partial \mathcal{L} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}\tag{9}$$
# $$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} \tag{10}$$
# 

# <div class="alert alert-info"><h2>Exercise 7</h2>
#     <p>Use the given 3 formulas above and complete the implemention of the function: <code>linear_backward</code> below.</p>
# </div>

# In[14]:


# GRADED FUNCTION: linear_backward

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


# In[15]:


# test your function below and compare your result to the expected output


np.random.seed(1)
dZ = np.random.randn(1,2)
A = np.random.randn(3,2)
W = np.random.randn(1,3)
b = np.random.randn(1,1)
linear_cache = (A, W, b)


#dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# **Expected Output**: 
# ```python
# dA_prev = [[ 0.51822968 -0.19517421]
#  [-0.40506361  0.15255393]
#  [ 2.37496825 -0.89445391]]
# dW = [[-0.10076895  1.40685096  1.64992505]]
# db = [[ 0.50629448]]
# ```

# ### 6.2 - Linear-Activation backward
# 
# Next, we'll merge the two helper functions `linear_backward` and `linear_activation_backward`. To ease this, a bit, we've provided two `backward` functions:
# 1. **`sigmoid_backward`**: which implements the backprop for the Sigmoid function. Here's how to use it:
# ```python
# dZ = sigmoid_backward(dA, Z_cache)
# ```
# 1. **`relu_backward`**: which implements backprop for the ReLU function. Here's how to use it:
# ```python
# dZ = relu_backward(dA, Z_cache)
# ```
# 
# Considering that the $g(.)$ is the activation function, use `sigmoid_backward` and `relu_backward` to compute $$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]}) \tag{11}$$.  
# 
# <div class="alert alert-info"><h2>Exercise 8</h2>
#     <p> Complete the below implementation of backprop for the <code>linear -> activation</code> layer. </p>
# </div>

# In[16]:



def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    np.reshape(dA,(1,np.product(dA.shape)))
    np.reshape(s,(1,np.product(s.shape)))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ





# GRADED FUNCTION: linear_activation_backward

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        
        ### START CODE HERE ### (~2 lines of code)
        
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        
        ### START CODE HERE ### (~2 lines of code)
        
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

        ### END CODE HERE ###
    
    return dA_prev, dW, db


# In[17]:


#  Test your function below and compare your result to the expected output

np.random.seed(2)
dA = np.random.randn(1,2)
A = np.random.randn(3,2)
W = np.random.randn(1,3)
b = np.random.randn(1,1)
Z = np.random.randn(1,2)
linear_cache = (A, W, b)
activation_cache = Z
linear_activation_cache = (linear_cache, activation_cache)

dA_prev, dW, db = linear_activation_backward(dA, linear_activation_cache, activation="sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(dA, linear_activation_cache, activation="relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# **Expected output:**
# ```python
# sigmoid:
# dA_prev = [[ 0.11017994  0.01105339]
#  [ 0.09466817  0.00949723]
#  [-0.05743092 -0.00576154]]
# dW = [[ 0.10266786  0.09778551 -0.01968084]]
# db = [[-0.05729622]]
# 
# relu:
# dA_prev = [[ 0.44090989  0.        ]
#  [ 0.37883606  0.        ]
#  [-0.2298228   0.        ]]
# dW = [[ 0.44513824  0.37371418 -0.10478989]]
# db = [[-0.20837892]]
# ```

# ### 6.3 - L-Model Backward 
# 
# Now, we'll implement the backward function for the whole network. When you implemented `L_model_forward`, at each iteration there was a `cache` of `(X, W, b, z)`. In the backprop module, we'll use those variables to compute the derivatives (also called gradients in the NN literature); therefore, in `L_model_backward`, you'll iterate through all the hidden layers backward, starting from layer $L$. At each step, you'll use the cached values for layer $l$ to backprop through layer $l$. Look below, at Figure 5, to see a backward pass.
# 
# <img src="images/mn_backward.png" style="width:450px;height:300px;">
# <caption><center>  **Figure 5** : Backward pass  </center></caption>
# 
# ** Initializing backpropagation**:
# To backpropagate through this network, we know that the output is, 
# $A^{[L]} = \sigma(Z^{[L]})$. Which means your code needs to compute `dAL` $= \frac{\partial \mathcal{L}}{\partial A^{[L]}}$.
# To do so, use this formula (derived using calculus which you don't need in-depth knowledge of):
# ```python
# dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
# ```
# 
# Then, this post-activation gradient (`dAL`) can be used to keep going backward. As seen in Figure 5, you can now feed `dAL` into the `linear -> sigmoid` backward function, which will use the cached values stored by `L_model_forward`). Afterwards, you'll have to use a `for` loop to iterate throughall the other layers using `linear -> relu`'s backward function. At each layer, you should store `dA`, `dW`, and `db` in the `grads` dict. To standardize things, use this format:
# 
# $$grads["dW" + str(l)] = dW^{[l]}\tag{15} $$
# 
# 
# For example, for $l=2$, this would store $dW^{[l]}$ in `grads["dW2"]`.
# 
# <div class="alert alert-info"><h2>Exercise 9</h2>
#     <p> Implement backprop for <code>([linear -> relu]</code> x <code>(L-1)) -> linear -> sigmoid</code> model. </p>
# </div>

# In[18]:


# GRADED FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    
    ### START CODE HERE ### (1 line of code)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    ### END CODE HERE ###
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    
    ### START CODE HERE ### (~2 lines)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
       
    ### START CODE HERE ### (~5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads


# In[19]:


def print_grads(grads):
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dA1 = "+ str(grads["dA1"]))     
    


def L_model_backward_test_case():
    """
    X = np.random.rand(3,2)
    Y = np.array([[1, 1]])
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}

    aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
           [ 0.02738759,  0.67046751],
           [ 0.4173048 ,  0.55868983]]),
    np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
    np.array([[ 0.]])),
   np.array([[ 0.41791293,  1.91720367]]))])
   """
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches



AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)


# **Expected Output**
# ```python
# dW1 = [[ 0.41010002  0.07807203  0.13798444  0.10502167]
#  [ 0.          0.          0.          0.        ]
#  [ 0.05283652  0.01005865  0.01777766  0.0135308 ]]
# db1 = [[-0.22007063]
#  [ 0.        ]
#  [-0.02835349]]
# dA1 = [[ 0.12913162 -0.44014127]
#  [-0.14175655  0.48317296]
#  [ 0.01663708 -0.05670698]]
# ```

# ### 6.4 - Update Parameters
# 
# In this section you will update the parameters of the model, using gradient descent: 
# 
# $$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{16}$$
# $$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{17}$$
# 
# where $\alpha$ is the learning rate. After computing the updated parameters, store them in the parameters dictionary. 

# <div class="alert alert-info"><h2>Exercise 10</h2>
# <p> Complete the implementation of <code>update_parameters</code> to update your parameters with gradient descent.</p>
# 
# <p><strong>Instructions:</strong> Update all the parameters: $W^{[l]}$ and $b^{[l]}$ for $l = 1, 2, ..., L$ by using gradient descent algorithm. </p>
# </div>

# In[20]:


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (~3 lines of code)
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    ### END CODE HERE ###
    return parameters


# In[21]:


def update_parameters_test_case():
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return parameters, grads



parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.01)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))


# **Expected Output**:
# ```python
# W1 = [[-0.43464413 -0.06063193 -2.13716107  1.65890574]
#  [-1.7906617  -0.83819978  0.50370883 -1.23901808]
#  [-1.05751404 -0.90423543  0.56459269  2.28336179]]
# b1 = [[ 0.03272621]
#  [-1.13502118]
#  [ 0.53855798]]
# W2 = [[-0.59211293 -0.0136769   1.19046599]]
# b2 = [[-0.75769462]]
# ```

# ## 7 - Conclusion
# 
# <div class="alert alert-success">
#     <p>You implemented forward and backward computations for L-layer neural networks! </p>
# </div>
# 
# Please switch to the part 2 now (file: `02_applying-a-dnn`) to use these implementations in cat classification.
# 
# <div class="alert alert-danger">
# <p><code>02_applying-a-dnn</code> involves combining all these functions to build two models:</p>
# <ol>
#     <li>a 2-layer neural network</li>
#     <li>an L-layer neural network</li>
# </ol>
# </div>
# 
# We'll actually be using these models to classify images as being cat or non-cat!

# In[22]:


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


# In[ ]:





# In[ ]:





# In[ ]:




