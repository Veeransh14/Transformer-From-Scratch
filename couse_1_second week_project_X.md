# Transformer From Scratch 


# Course 1-
# Week 2-

# Binary Classification
Binary classification basically involves with working which two values 1 and 0 where 1 denotes true condition and zero denotes false condition example if there is a picture of cat model has to identify whether that picture is really of a cat or of any other animal hence if our model detects that this picture is of a cat then it returns a value 1 else it will return a value of zero this is the basic concept of binary classification.

In a similar way this is possible for a pixel where we have three colours matrix of red blue and green respectively having dimension 64 by 64 and hence the total number of dimensions would be 64 by 64 into 3(because of three colours)output is the algebraic sum of all the values returned in these dimensions the resultant vector produced by these calculations give us the output which is denoted by letter y.
We can see the same in the picture given below.
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/ee8edfed-7975-4cde-8d27-e1acc04cf41c)


# Notation 
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/1219884b-eae3-4325-8c12-4d1ca0b6dda9)

We know what is (x,y) very well here we will create a matrix X of order nXm where m denites number of training examples (columns) and n denotes the dimensional feature vector. y can have value={0,1}
We would be having M(train) and M(test) function present in our model. 
X slope=(n,m).

![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/4437140c-e4e4-4bc9-b6dc-3772d0196852)

# Logistic Regression
Logistic regression is popularly used in binary classification as this algorithm is a learning algorithm and it works with binary operations 0 and 1.We know that in binary classification the output received that is Zero OR one that is it is either false or true logistic regression mainly works on this principle.

![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/023a67d6-5c21-4b6c-b9a2-46a8854e8aa5)

We use sigmoid function because it operates between values 0 and 1 which is required by us in binary classification.
OUTPUT- y=sigmoid function(w*x+b) where b belongs to real number

![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/e2f79477-0110-47ec-b7ad-b562b3b991ef)

# Cost Function
The cost function is the average error of n-samples in the data (for the whole training data) and the loss function is the error for individual data points (for one training example).
The cost function of a linear regression is root mean squared error or mean squared error. They are both the same; just we square it so that we don’t get negative values.
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/06d64196-7c84-4a92-8967-b29f1e149bce)

# Loss Function
The process of learning from data to find the solution to a problem is machine learning. Ideally, the dataset we find has labels making it a supervised problem. The learning process is all about using the training data to produce a predictor function. It maps input vector ‘x’ to ground truth ‘y’. We want the predictor to work even for examples that it hasn’t yet seen before in the training data. That is, we want it to be as generalized as possible. And because we want it to be general, this forces us to design it in a principled, mathematical way.
For each data point ‘x’, we start computing a series of operations on it. Whatever operations our model requires to produce a predicted output, we then compare the predicted output to the actual output ‘y’ to produce an error value. That error is what we minimize during the learning process using an optimization strategy like gradient descent.
We are computing that error value by using a loss function. Loss functions can calculate errors associated with the model when it predicts ‘x’ as output and the correct output is ‘y’*.
Unfortunately, there is no universal loss function that works for all kinds of data. There are many factors that affect the decision of which loss function to use like the outliers, the machine learning algorithm, speed-accuracy trade-off, etc.
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/ba6bb1bf-31f8-48fe-97a9-a5ad315c2092)

# Code for Logistic Regression
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/433ea482-4047-4318-8a37-0e9bc0b41f19)

# Gradient Descent
In Supervised Learning a machine learning algorithm builds a model which will learn by examining multiple examples and then attempting to find out a function which minimizes loss. Since we are unaware of the true distribution of the data on which the algorithm will work so we instead measure the performance of the algorithm on known set of data i.e Training dataset. This process is known as Empirical Risk Minimization.
Loss is the measures how well the algorithm is performing on the given dataset. Thus loss is a number which indicates how bad model prediction was on a single example. If the model prediction is 100% accurate then the loss will be zero else the loss will be greater. The function responsible for calculating the penalty is generally referred to as Cost Function.

Gradient Descent in simple terms is an algorithm which minimizes a function. The general idea of Gradient Descent is to tweak parameters(weights and biases) iteratively in order to minimize a cost function. Suppose you are standing at the top of the mountain and you want to get down to the bottom of the valley as quickly as possible , the good strategy is to go downhill in the direction of steepest slope. This is exactly how gradient descent works, it measures the local gradient of the loss function with regards to parameterand it goes in the direction of negative gradient. Once the gradient is zero, you have reached the minimum.
Concretely,start by initializing with a random value and then improve it gradually taking one small step at a time,each step attempting to decrease the cost function(eg: MSE),until algorithm converges to minimum.
With a random value and then improve it gradually taking one small step at a time , each step attempting to decrease the cost function.

![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/3180597d-f412-410b-a7c7-85e2c817aba5)


# Derivatives
Derivatives are a fundamental concept in calculus, and they play a crucial role in many machine-learning algorithms. Put simply, a derivative measures the rate of change of a function at a particular point. This information can be used to optimize functions, find local minima and maxima, and more.
We can know the nature of graph with help of derivatives and hence predict the outcome.The slope changes at each point hence the nature of graph changes. This helps us to compute error and makes us reduce with time.
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/bd0f1e31-d72a-4f6a-a9f0-d5019d23de5c)

# Computational Graphs
A computational graph is defined as a directed graph where the nodes correspond to mathematical operations. Computational graphs are a way of expressing and evaluating a mathematical expression.

For example, here is a simple mathematical equation −

p=x+y
We can draw a computational graph of the above equation as follows.
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/884000cd-4da4-4ccd-9139-70adb2adc710)

Computational Graph Equation1
The above computational graph has an addition node (node with "+" sign) with two input variables x and y and one output q.

g=(x+y)∗z
The above equation is represented by the following computational graph.
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/9366dd24-47a2-4bcd-acc3-16db26fa2883)


# Computational Graphs and Backpropagation
Computational graphs and backpropagation, both are important core concepts in deep learning for training neural networks.

Forward Pass
Forward pass is the procedure for evaluating the value of the mathematical expression represented by computational graphs. Doing forward pass means we are passing the value from variables in forward direction from the left (input) to the right where the output is.

Let us consider an example by giving some value to all of the inputs. Suppose, the following values are given to all of the inputs.

x=1,y=3,z=−3
By giving these values to the inputs, we can perform forward pass and get the following values for the outputs on each node.

First, we use the value of x = 1 and y = 3, to get p = 4.
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/4b11d1f2-3b9e-4a00-aad1-52ba641161a5)

Forward Pass
Then we use p = 4 and z = -3 to get g = -12. We go from left to right, forwards.
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/0bf05eb9-e3e5-409b-8745-629cc0457702)


# Objectives of Backward Pass
In the backward pass, our intention is to compute the gradients for each input with respect to the final output. These gradients are essential for training the neural network using gradient descent.

For example, we desire the following gradients.

Desired gradients
∂x∂f,∂y∂f,∂z∂f
Backward pass (backpropagation)
We start the backward pass by finding the derivative of the final output with respect to the final output (itself). Thus, it will result in the identity derivation and the value is equal to one.

∂g∂g=1
Our computational graph now looks as shown below
![image](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/723bab6f-cfe0-456f-8d81-6153fee5c728)

The main reason for doing this backwards is that when we had to calculate the gradient at x, we only used already computed values, and dq/dx (derivative of node output with respect to the same node's input). We used local information to compute a global value.

# Steps for training a neural network
Follow these steps to train a neural network −
1.For data point x in dataset,we do forward pass with x as input, and calculate the cost c as output.
2.We do backward pass starting at c, and calculate gradients for all nodes in the graph. This includes nodes that represent the neural network weights.
3.We then update the weights by doing W = W - learning rate * gradients.
4.We repeat this process until stop criteria is met.


