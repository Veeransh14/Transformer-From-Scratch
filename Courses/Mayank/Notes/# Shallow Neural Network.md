# Shallow Neural Network

A shallow neural network has only one (or just a few) hidden layers between the input and output layers. The input layer receives the data, the hidden layer processes it and the final layer produces the output.
Shallow neural networks are simpler, more easily trained, and have greater computational efficiency than deep neural networks.

## Learning Objectives
- Describe hidden units and hidden layers
- Use units with a non-linear activation function, such as tanh
- Implement forward and backward propagation
- Apply random initialization to your neural network
- Increase fluency in Deep Learning notations and Neural Network Representations
- Implement a 2-class classification neural network with a single hidden layer
- Compute the cross entropy loss

## Table of Content
- 1. Neural Network Overview and Representation
- 2. Computing a Neural Networks Output 
- 3. Vectorizing across Multiple examples
- 4. Why do You need Non-Linear Activation Functions?
- 5. Derivatives of Activation Functions
- 6. Gradient Decesent in Neural Networks
     - Forward Propogation
     - Backward Propogation
- 7. Random Initialization

#### Neural Network Overview and Representation
This is what Logistic Regression looks like
![](https://hackmd.io/_uploads/Hk6RTGhRh.png)
A neural network can be formed by stacking together a lot of little sigmoid units. 
A neural network looks like
![](https://hackmd.io/_uploads/BJx00zhA2.png)

The 'X1, X2, X3' in the given representation are the input layers. The layers which come after the input layer is the hidden layer, which evaluates/ compiles and gives us the output layer which is termed as 'Y(hat)'.

#### Computing a Neural Networks Output
Computing Neural Network's ouput occurs in three phases: 
1. The first phase is to deal with raw input values.
2. The second phase is to compute the values for the hidden-layer nodes 
3. The third phase is to compute the values for output-layer nodes.

Equation for the hidden layers:

![](https://aman.ai/coursera-dl/assets/neural-networks-and-deep-learning/05.png)

- Input size (Nx)=3
- No of Hidden Neurons (Hn)= 4
- Shapes of the Variables:
     - W1 is the matrix of the first hidden layer, the size of the matrix = (Hn,Nx)
     
     - b1 is the matrix of the first hidden layer, the size of the matrix = (Hn,1)
     - Z1 is the result of the equation Z1 = W1*X + b, it has matrix size of (Hn,1)
     - a1 is the result of the equation a1 = sigmoid(z1), it has matrix size of (Hn,1)
     - W2 is the matrix of the second hidden layer, it has a size of (1,noOfHiddenNeurons)
     - b2 is the matrix of the second hidden layer, it has a size of (1,1)
     - z2 is the result of the equation z2 = W2*a1 + b, it has matrix size of (1,1)
     - a2 is the result of the equation a2 = sigmoid(z2), it has matrix size of (1,1)

#### Vectorizing across multiple examples
Pseudo code for forward propagation for the 2 layers NN:
`for i = 1 to m
  z[1, i] = W1*x[i] + b1      # shape of z[1, i] is (noOfHiddenNeurons,1)
  a[1, i] = sigmoid(z[1, i])  # shape of a[1, i] is (noOfHiddenNeurons,1)
  z[2, i] = W2*a[1, i] + b2   # shape of z[2, i] is (1,1)
  a[2, i] = sigmoid(z[2, i])  # shape of a[2, i] is (1,1)`

Lets say we have X on shape (Nx,m). So the new pseudo code:
`Lets say we have X on shape (Nx,m). So the new pseudo code:`
In the last example we can call X = A0. So the previous step can be rewritten as:
`In the last example we can call X = A0. So the previous step can be rewritten as:`

#### Activation Function
So far we have used Sigmoid, but in some cases we have to use other functions for better outputs. Sigmoid function can lead us to gradient descent problem where the updates are so low. 
Sigmoid Activation function range is [0,1] 
__A = 1 / (1 + np.exp(-z)) # Where z is the input matrix__

Tanh activation function range is [-1,1]
__In NumPy we can implement Tanh using one of     these methods:` A = (np.exp(z) - np.exp(-z)) /     (np.exp(z) + np.exp(-z)) # Where z is the input    matrix Or A = np.tanh(z)` # __Where z is the         input matrix__

The tanh activation function works much better than sigmoid activation function for hidden units because the mean of its output is closer to 0, and so it can center the data better for the next layer. 

The disadvantage of sigmoid or tanh function is that if the input is too small or too high then the slope will be near zero which will cause us the gradient descent problem.

Relu is one of the popular activation functions that solved the slow gradient descent   
__RELU = max(0,z), so if z is negative the slope is 0 and if z is positive the slope remains linear.__

So here is some basic rule for choosing activation functions, if your classification is between 0 and 1, use the output activation as sigmoid and the others as RELU.
Leaky RELU activation function different of RELU is that if the input is negative the slope will be so small. It works as RELU but most people uses RELU. Leaky_RELU = max(0.01z,z), the 0.01 can be a parameter for your algorithm.

In NN you will decide a lot of choices like:
No of hidden layers.
No of neurons in each hidden layer.
Learning rate. (The most important parameter)
Activation functions.


#### Why do You need Non-Linear Activation Functions?
- If we removed the activation function from our algorithm that can be called linear activation function.
- Linear activation function will output linear activations
- Whatever hidden layers you add, the activation will be always linear like logistic regression (So its useless in a lot of complex problems)
- You might use linear activation function in one place - in the output layer if the output is real numbers (regression problem). But even in this case if the output value is non-negative you could use RELU instead.


#### Derivatives of Activation Functions 
Derivation of Sigmoid activation function:
`g(z)  = 1 / (1 + np.exp(-z))
g'(z) = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
g'(z) = g(z) * (1 - g(z))`

Derivation of Tanh activation function:
`g(z)  = (e^z - e^-z) / (e^z + e^-z)
g'(z) = 1 - np.tanh(z)^2 = 1 - g(z)^2`

Derivation of Relu activation function:
`g(z)  = np.maximum(0,z)
g'(z) = { 0  if z < 0
          1  if z >= 0  }`
         
Derivation of leaky Relu activation function:
`g(z)  = np.maximum(0.01 * z, z)
g'(z) = { 0.01  if z < 0
          1     if z >= 0   }`


#### Gradient Decesent in Neural Networks 
Gradient Descent is known as one of the most commonly used optimization algorithms to train machine learning models by means of minimizing errors between actual and expected results. Further, gradient descent is also used to train Neural Networks.

In mathematical terminology, Optimization algorithm refers to the task of minimizing/maximizing an objective function f(x) parameterized by x. Similarly, in machine learning, optimization is the task of minimizing the cost function parameterized by the model's parameters. The main objective of gradient descent is to minimize the convex function using iteration of parameter updates. Once these machine learning models are optimized, these models can be used as powerful tools for Artificial Intelligence and various computer science applications.

- __Forward Propogation:__

In Forward Propogation, the input data is fed in the forward direction only. The data should not flow in the reverse direction during output generation otherwise it would form a cycle and the output could never be generated. Such network configuration are known as feed-forward network. The feed-forward network helps in forward propogation.

- __Backward Propogation:__

Backward propogation is a process involved in training a neural network. It involves taking the error rate of a forward propogation and feeding this loss backward through the neural network layers to fine-tune the weights.

#### Random Initialization
In Neural Networks we have to initialize weights randomly.
If we initialize all the weights with zeros in NN it won’t work (initializing bias with zero is OK):
all hidden units will be completely identical (symmetric) - compute exactly the same function
on each gradient descent iteration all the hidden units will always update the same.
We need small values because in sigmoid (or tanh), for example, if the weight is too large you are more likely to end up even at the very start of training with very large values of Z. Which causes your tanh or your sigmoid activation function to be saturated, thus slowing down learning. If you don’t have any sigmoid or tanh activation functions throughout your neural network, this is less of an issue.

Constant 0.01 is alright for 1 hidden layer networks, but if the NN is deep this number can be changed but it will always be a small number.

