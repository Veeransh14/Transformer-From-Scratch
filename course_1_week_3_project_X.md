# Transformer From Scratch

# Course  1-
# Week 3-

# Computing Neural Networks
Steps to Perform Neural Network
There are three steps to perform in any neural network:
1.We take the input variables and the above linear combination equation of  Z = W0 + W1X1 + W2X2 + …+ WnXn to compute the output or the predicted Y values, called the Ypred.
2.Calculate the loss or the error term. The error term is the deviation of the actual values from the predicted values.
3.Minimize the loss function or the error term.
![Alt text](image-18.png)



# How to Calculate the Output for a Neural Network using vectorized approach?
Let's consider the example given below-
![Alt text](image-19.png)
Firstly, we will understand how to calculate the output for a neural network and then will see the approaches that can help to converge to the optimum solution of the minimum error term.

The output layer nodes are dependent on their immediately preceding hidden layer L3, which is coming from the hidden layer 2 and those nodes are further derived from the input variables. These middle hidden layers create the features that the network automatically creates and we don’t have to explicitly derive those features. In this manner, the features are generated in Deep Learning models and this is what makes them stand out from Machine Learning.

So, to compute the output, we will have to calculate for all the nodes in the previous layers. Let us understand what is the mathematical explanation behind any kind of neural nets.

Now, as from the above architecture, we can see that each neuron cannot have the same general equation for the output as the above one. We will have one such equation per neuron both for the hidden and the output layer.

The nodes in the hidden layer L2 are dependent on the Xs present in the input layer therefore, the equation will be the following:

N1 = W11*X1 + W12*X2 + W13*X3 + W14*X4 + W10
N2 = W21*X1+ W22*X2 + W23*X3 + W24*X4 + W20
N3 = W31*X1+ W32*X2 + W33*X3 + W34*X4 + W30
N4 = W41*X1+ W42*X2 + W43*X3 + W44*X4 + W40
N5 = W51*X1+ W52*X2 + W53*X3 + W54*X4 + W50
Similarly, the nodes in the hidden layer L3 are derived from the neurons in the previous hidden layer L2, hence their respective equations will be:

N5 = W51 * N1 + W52 * N2 + W53 * N3 + W54 * N4 + W55 * N5 + W50
N6 = W61 * N1 + W62 * N2 + W63 * N3 + W64 * N4 + W65 * N5 + W60
N7 = W71 * N1 + W72 * N2 + W73 * N3 + W74 * N4 + W75 * N5 + W70
The output layer nodes are coming from the hidden layer L3 which makes the equations as:

O1 = WO11 * N5 + WO12 * N6 + WO13 * N7 + WO10
O2 = WO21 * N5 + WO22 * N6 + WO23 * N7 + WO20

Now, how many weights or betas will be needed to estimate to reach the output? On counting all the weights Wis in the above equation will get 51. However, no real model will have only three input variables to start with!

Additionally, the neurons and the hidden layers themselves are the tuning parameters so in that case, how will we know how many weights to estimate to calculate the output? Is there an efficient way than the manual counting approach to know the number of weights needed? The weights here are referred to the beta coefficients of the input variables along with the bias term as well (and the same will be followed in the rest of the article).

The structure of the network is 4,5,3,2. The number of weights for the hidden layer L2 would be determined as = (4 + 1) * 5 = 25, where 5 is the number of neurons in L2 and there are 4 input variables in L1. Each of the input Xs will have a bias term which makes it 5 bias terms, which we can also say as (4 + 1) = 5.

Therefore, the number of weights for a particular layer is computed by taking the product of (number of nodes/variables + bias term of each node) of the previous layer and the number of neurons in the next layer

Similarly, the number of weight for the hidden layer L3 = (5 + 1) * 3 = 18 weights, and for the output layer the number of weights = (3 + 1) * 2 = 8.

The total number of weights for this neural network is the sum of the weights from each of the individual layers which is = 25 + 18 + 8 = 51

We now know how many weights will we have in each layer and these weights from the above neuron equations can be represented in the matrix form as well. Each of the weights of the layers will take the following form:

Hidden Layer L2 will have a 5 * 5 matrix as seen the number of weights is (4 + 1) * 5:

N1 = W11*X1 + W12*X2 + W13*X3 + W14*X4 + W10
N2 = W21*X1+ W22*X2 + W23*X3 + W24*X4 + W20
N3 = W31*X1+ W32*X2 + W33*X3 + W34*X4 + W30
N4 = W41*X1+ W42*X2 + W43*X3 + W44*X4 + W40
N5 = W51*X1+ W52*X2 + W53*X3 + W54*X4 + W50
Estimation of Neurons hidden layer 2
A 3*6 matrix for the hidden layer L3 having the number of weights as (5 + 1) * 3 = 18

N5 = W51 * N1 + W52 * N2 + W53 * N3 + W54 * N4 + W55 * N5 + W50
N6 = W61 * N1 + W62 * N2 + W63 * N3 + W64 * N4 + W65 * N5 + W60
N7 = W71 * N1 + W72 * N2 + W73 * N3 + W74 * N4 + W75 * N5 + W70
3*6 matrix Estimation of Neurons
Lastly, the output layer would be 4*2 matrix with (3 + 1) * 2 number of weights:

O1 = WO11 * N5 + WO12 * N6 + WO13 * N7 + WO10
O2 = WO21 * N5 + WO22 * N6 + WO23 * N7 + WO20
Estimation of Neurons matrix
Okay, so now we know how many weights we need to compute for the output but then how do we calculate the weights? In the first iteration, we assign randomized values between 0 and 1 to the weights. In the following iterations, these weights are adjusted to converge at the optimal minimized error term.

We are so persistent about minimizing the error because the error tells how much our model deviates from the actual observed values. Therefore, to improve the predictions, we constantly update the weights so that loss or error is minimized.

This adjustment of weights is also called the correction of the weights. There are two methods: Forward Propagation and Backward Propagation to correct the betas or the weights to reach the convergence. We will go into the depth of each of these techniques; however, before that lets’ close the loop of what the neural net does after estimating the betas.

# Activation functions
![Alt text](image-20.png)
Like above examples we use many other Activation function to implement vectorization however our preference of use of a function mainly depends upon the type of data its dealing with and its nature.
We use many different activation functions depending upon the data set and also the algorithm to some extent the data set primarily depends upon what are model is performing and what is its function generally we don't use tanh function  this is because of the fact that its values ranges from -1 to 1 and in many cases were required output values that is output to be from 0 and 1. This problem forces us to use sigmoid function which gives output from 0 to 1. However in certain cases even sigmoid function is not used because for higher values of x the slope and tends to zero due to this negligible value of slope it difficult to back propagate and chances of error also increases the speed with which are model is trained is decreased considerably. so instead of using sigmoid nowadays we use relu and leeky relu functions  as shown above in the figure. Many of the algorithm are based on these functions.

 # Whats the need of non linear Activation functions?
 We use non linear activation functions primarily because it is observed that the data set which are model is dealing is not linear in majority of the cases. If use then it becomes difficult for our model to get train and deal with nonlinear values of data.   On other hand if our function is non linear in nature then it becomes very easy for our model to train and adapt to new things for using nonlinear activation functions. 

Derivatives for Activation function can be easily calculated using our basic calculus knowledge this helps us in later stage to calculate dW,dZ etc.

# Gradient Descent
Gradient Descent is known as one of the most commonly used optimization algorithms to train machine learning models by means of minimizing errors between actual and expected results. Further, gradient descent is also used to train Neural Networks.
In mathematical terminology, Optimization algorithm refers to the task of minimizing/maximizing an objective function f(x) parameterized by x. Similarly, in machine learning, optimization is the task of minimizing the cost function parameterized by the model's parameters. The main objective of gradient descent is to minimize the convex function using iteration of parameter updates. Once these machine learning models are optimized, these models can be used as powerful tools for Artificial Intelligence and various computer science applications.
The main objective of using a gradient descent algorithm is to minimize the cost function using iteration.
![Alt text](image-21.png)

# Back Propogation
You already know that we randomly initialize the weights and biases in the network and use that to make predictions, just like how we randomly initialized x in the previous section. we take these predictions made by our neural network and use some sort of metric to measure the deviation between the actual target and our model’s output ( This is nothing but the loss function ).

We then proceed to differentiate the loss function with respect to each and every individual weight in the network, similar to how we differentiated y with respect to x in the previous example. Once this is done, we update the weights in the direction opposite to the differentiated term..this is called backpropogation.

Objective: To find the derivatives for the loss or error with respect to every single weight in the network, and update these weights in the direction opposite to their respected derivatives so as to move towards the global or local minima of the cost or error function.

Before we begin, One special feature of the sigmoid activation function is that its derivative is very easy to calculate.
derivative of sigmoid(x) = sigmoid(x) * (1 — sigmoid(x)).
![Alt text](image-22.png)

![Alt text](image-23.png)

# Random Initialization
Random Initialization for neural networks aids in the symmetry-breaking process and improves accuracy. The weights are randomly initialized in this manner, very close to zero. As a result, symmetry is broken, and each neuron no longer performs the same computation. Usually it's as small as 0.01