# Deep Neural Network
Deep Neural Networks is a class of machine learning algorithms similar to artificial Neural Network and aims to mimic the information processing of the brain.

Table Of Content:
- 1. Deep L-layer Neural Network
- 2. Forward Propogation in a Deep Neural Network
- 3. Getting the Matrix Dimensions Right
- 4. Why Deep Representation?
- 5. Forward and Backward Propogation
- 6. Parameters vs Hyperparameters

#### Deep L-layer Neural Network
Shallow Neural Netwrok is a neural network with one or two layers. Deep Neural Network is a neural network with three or more layers. We use the notation L to denote the number of layers in a Neural network.
n[0] denotes the number of layers in input layer, n[1] denotes the number of layers in output layer.
g[l] is the activation function.
a[l] = g[l].z[l]
w[l] weights is used for z[l]
x = a[0], a[l] = y'

We have:
A vector n of shape (1, NoOfLayers+1)
A vector g of shape (1, NoOfLayers)
A list of different shapes w based on the number of neurons on the previous and the current layer.
A list of different shapes b based on the number of neurons on the current layer.

#### Forward Propogation in a Deep Neural Network
Forward propagation is where input data is fed through a network, in a forward direction, to generate an output. The data is accepted by hidden layers and processed, as per the activation function, and moves to the successive layer. The forward flow of data is designed to avoid data moving in a circular motion, which does not generate an output. 

![](https://hackmd.io/_uploads/SJ01Lb1y6.png)

#### Getting the Matrix Dimensions Right
The best way to debug your matrices dimensions is by a pencil and paper.
Dimension of W is (n[l],n[l-1]) . Can be thought by right to left.
Dimension of b is (n[l],1)
dw has the same shape as W, while db is the same shape as b
Dimension of Z[l], A[l], dZ[l], and dA[l] is (n[l],m)

#### Why Deep Representation?
We’ve heard that neural networks work really well for a lot of problems. However, neural networks doesn’t need only to be big. Neural Networks also need to be deep or to have a lot hidden layers.

If we are, for example, building a system for an image classification, here is what a deep neural network could be computing. The input of a neural network is a picture of a face. The first layer of the neural network could be a feature detector, or an edge detector. So, the first layer can look at the pictures and find out where are the edges in the picture. Then, in next layer those detected edges could be grouped together to form parts of faces. By putting a lot of edges it can start to detect different parts of faces. For example, we might have a low neurons trying to see if it’s finding an eye or a different neuron trying to find  part of a nose. Finally, putting together eyes, nose etc. it can recognise different faces. 

![](https://hackmd.io/_uploads/rkHnI-Jyp.png)


To conclude, earlier layers of a neural network detects simpler functions (like edges), and composing them together, in the later layers of a neural network, deep neural network can compute more complex functions.


#### Forward and Backward Propogation

Forward and Backward Propagation

Pseudo code for forward propagation for layer l:

`Input  A[l-1]

    Z[l] = W[l]A[l-1] + b[l]
    A[l] = g[l](Z[l])
    Output A[l], cache(Z[l])`

Pseudo code for back propagation for layer l:

`Input da[l], Caches    
    
    dZ[l] = dA[l] * g'[l](Z[l])
    dW[l] = (dZ[l]A[l-1].T) / m
    db[l] = sum(dZ[l])/m                
    dA[l-1] = w[l].T * dZ[l]            
    Output dA[l-1], dW[l], db[l]`

If we have used our loss function then:


`dA[L] = (-(y/a) + ((1-y)/(1-a)))`

#### Parameters vs Hyperparameters
- Main parameters of the NN is W and b
- Hyper parameters (parameters that control the algorithm) are like:
- Learning rate.
- Number of iteration.
- Number of hidden layers L.
- Number of hidden units n.
- Choice of activation functions.
- You have to try values yourself of hyper parameters.
- In the earlier days of DL and ML learning rate was often called a parameter, but it really is (and now everybody call it) a hyperparameter.


