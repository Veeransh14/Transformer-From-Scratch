## Logistic Regression as a Neural Network

Welcome to the first(required) programming assignment. This assignment will set up a machine learning problem with a neural network mindset and use vectorization to speed up the models. 

### You will learn to:
- Build a logistic regression model structured as a shallow neural network
- Build the general architecture of a learning algorithm, including parameter initialization, cost function and gradient calculation, and optimization implemetation (gradient descent)
- Implement computationally efficient and highly vectorized versions of models
- Compute derivatives for logistic regression, using a backpropagation mindset
- Use Numpy functions and Numpy matrix/vector operations
- Work with iPython Notebooks
- Implement vectorization across multiple training examples
- Explain the concept of broadcasting

### Table of Contents
- 1. Binary Classification
- 2. Logistic Regression
     - Logistic Function(Sigmoid Function)
     - Types of Logistic Regression 
     - How do Logistic Regression Work?
     - Sigmoid Function
     - Logistic Regression Equation

- 3. Logistic Regression Cost Function
     - Cost Function
     - Loss Function
- 4. Gradient Descent
- 5. Derivatives 
- 6. Computational Graph
     - Computational Graph and Backpropogation
- 7. Vectorization
     - Short Instruction/ Multiple Data
- 8. Vectorizing Logistic Regression
- 9. Vectorizing Logistic Regression's Gradient Output
- 10. Broadcasting in Python


#### Binary Classification
One of the most common uses for machine learning is performing binary classification, which looks at an input and predicts which of the two possible classes it belongs to. The two answers are generally 0 and 1. 

To take an example let us take a problem statement:
![](https://hackmd.io/_uploads/rkulCS9C3.png)

We have a input of an image, and one output label as well. To recognise this image of it being either a cat for which the output would be 1(true), or a non-cat for which the output would be 0(false). We use 'Y' to denote the output label.
The image in a computer is stored by determining three different matrices corresponding to the red, green and blue color channels. 
Let us say that the image is 64 pixels by 64 pixels, then we would have three 64 by 64 matrices corresponding to the red, green and blue pixel intensity values, for the image.

To turn these pixel intensity values into a feature vector, we will unroll all of these pixel values into a input feature vector X, so to unroll all of these pixel intensity values into a feature vector, we will define a feature vector X corresponding to this image as:
 - It is going to take all the pixel values like 255 231..., 255 134 ... and so on, until we list all the red pixels. All these values are listed in a single column matrix(for one color).This is done for the pixels of all three colours.
- So if the image pixel is 64 by 64, then the total dimension of this vector X will be  64 * 64 * 3, which comes out as 12288 
-  n= 12288 (dimension inout of the image)

Binary classification is basically to take in the input 'X' and give out any of the two outputs (either 1 or 0) 'Y'.


#### Logistic Regression
Logistic regression is a supervised machine learning algorithm mainly used for classification tasks where the goal is to predict the probability that an instance of belonging to a given class. It is used for classification of algorithms.
Logostic regression is used for predicting categorical dependant variable using a given set of independant variables 

- Logistic regression predicts the output of a categorical variable, and hence the output must be a categorical or a discrete value. 
- It gives probabilistic values which lie between 0 to 1. 
- Logistic Regression is a significant machine learning algorithm because it has the ability to provide probabilities and classify new data using continuous and discrete datasets.

##### Logistic function (Sigmoid Function):
- The sigmoid function is a mathematical function used to map the predicted values to probabilities.
- The value of the logistic regression must be between 0 and 1, which cannot go beyond this limit, so it forms a curve like the 'S' form.
- The S-form curve is called the Sigmoid function or the logistic function.
- In logistic regression, we use the concept of the threshold value, which defines the probability of either 0 or 1. Such as values above the threshold value tends to 1, and a value below the threshold values tends to 0.

##### Types of Logistic Regression:
1. Binomial: In binomial Logistic regression, there can be only two possible types of the dependent variables, such as 0 or 1, Pass or Fail, etc.
2. Multinomial: In multinomial Logistic regression, there can be 3 or more possible unordered types of the dependent variable, such as “cat”, “dogs”, or “sheep”
3. Ordinal: In ordinal Logistic regression, there can be 3 or more possible ordered types of dependent variables, such as “low”, “Medium”, or “High”.

##### How do Logistic Regression Work?
The logistic regression model transforms the linear regression function continuous value output into categorical value output using a sigmoid function, which maps any real-valued set of independent variables input into a value between 0 and 1. This function is known as the logistic function.

Let the independent input features be

 X = \begin{bmatrix} x_{11}  & ... & x_{1m}\\ x_{21}  & ... & x_{2m} \\  \vdots & \ddots  & \vdots  \\ x_{n1}  & ... & x_{nm} \end{bmatrix} 

 and the dependent variable is Y having only binary value i.e. 0 or 1. 

Y = \begin{cases} 0 & \text{ if } Class\;1 \\ 1 & \text{ if } Class\;2 \end{cases}

then apply the multi-linear function to the input variables X

z = \left(\sum_{i=1}^{n} w_{i}x_{i}\right) + b  

Here x_i      is the ith observation of X, w_i = [w_1, w_2, w_3, \cdots,w_m]      is the weights or Coefficient, and b is the bias term also known as intercept. simply this can be represented as the dot product of weight and bias.

                            z = w\cdot X +b  

##### Sigmoid Function
Now we use the sigmoid function where the input will be z and we find the probability between 0 and 1. i.e predicted y.

![](https://hackmd.io/_uploads/B1XIOU50n.png)

![](https://hackmd.io/_uploads/B1OIuUqCh.png)


As shown above, the figure sigmoid function converts the continuous variable data into the probability i.e. between 0 and 1. 

![](https://hackmd.io/_uploads/S1axuUqRn.png)tends towards 1 as ![](https://hackmd.io/_uploads/rk-GOU5Rh.png)

![](https://hackmd.io/_uploads/r1-b_UcA2.png)tends towards 0 as ![](https://hackmd.io/_uploads/HJAMdUqC2.png)

![](https://hackmd.io/_uploads/BJV-_L9C3.png)   is always bounded between 0 and 1
where the probability of being a class can be measured as:

![](https://hackmd.io/_uploads/HJuyuUcAh.png)


##### Logistic Regression Equation
The odd is the ratio of something occurring to something not occurring. it is different from probability as the probability is the ratio of something occurring to everything that could possibly occur. so odd will be

![](https://hackmd.io/_uploads/H1dAvU50n.png)


Applying natural log on odd. then log odd will be

![](https://hackmd.io/_uploads/HJ5jv8902.png)

then the final logistic regression equation will be:

![](https://hackmd.io/_uploads/H1OTP85Rn.png)

#### Logistic Regression Cost Function
Cost function is basically an error representation in machine learning, it shows how our model is predicting compared to original given dataset. __Lower the cost function greater is the accuracy__.
Our main aim is to minimize the cost function.
We can represent the error by:
`|Y(predicted value)-Y(actual value)|`

Therefore the formula of cost function for Logistic Regression is:
![](https://hackmd.io/_uploads/BJhYr1o0n.png)

![](https://hackmd.io/_uploads/rJEtGXs0n.png)



__If y=0, error= -1*log(1-Y(pred))__ 
![](https://hackmd.io/_uploads/rJB1FksC3.png)

 - if Y(pred) is closer to 0 the error is less 
 - if Y(pred) is closer to 1 the error is more


__If y=1, error= -log(Y(pred))__ 
![](https://hackmd.io/_uploads/BkrROkjA3.png)

 - if Y(pred) is closer to 1 the error is less
 - if Y(pred) is closer to 0 the error is high

#### Loss Function 
The process of learning from data to find the solution to a problem is machine learning. Ideally, the dataset we find has labels making it a supervised problem. The learning process is all about using the training data to produce a predictor function. It maps input vector ‘x’ to ground truth ‘y’. We want the predictor to work even for examples that it hasn’t yet seen before in the training data. That is, we want it to be as generalized as possible. And because we want it to be general, this forces us to design it in a principled, mathematical way. For each data point ‘x’, we start computing a series of operations on it. Whatever operations our model requires to produce a predicted output, we then compare the predicted output to the actual output ‘y’ to produce an error value. That error is what we minimize during the learning process using an optimization strategy like gradient descent. We are computing that error value by using a loss function. Loss functions can calculate errors associated with the model when it predicts ‘x’ as output and the correct output is ‘y’*. Unfortunately, there is no universal loss function that works for all kinds of data.
![](https://hackmd.io/_uploads/rkKIegjCh.png)



#### Gradient Descent

Gradient descent is an optimization algorithm which is commonly used to train machine learning models and neural networks. Training the data helps these models learn over time, and the cost function within the gradient descent specifically acts as a barometer, gauging its accuracy with each iteration of the parameter updates. Until the function is close to or equal to zero, the model will continue to adjust its parameters to yield the smallest possible error.
__An algorthim to minimize a function by optimizing parameters__

The general idea of Gradient Descent is to tweak parameters(weights and biases) iteratively in order to minimize a cost function. Suppose you are standing at the top of the mountain and you want to get down to the bottom of the valley as quickly as possible , the good strategy is to go downhill in the direction of steepest slope. This is exactly how gradient descent works, it measures the local gradient of the loss function with regards to parameterand it goes in the direction of negative gradient. Once the gradient is zero, you have reached the minimum. Concretely,start by initializing with a random value and then improve it gradually taking one small step at a time,each step attempting to decrease the cost function(eg: MSE),until algorithm converges to minimum. With a random value and then improve it gradually taking one small step at a time , each step attempting to decrease the cost function.

![](https://hackmd.io/_uploads/rkwGbxsAh.png)



#### Derivatives
A derivative is a continuous description of how a function changes with small changes in one or more multiple variables. This information of derivatives can be used to optimize functions, find local minima and maxima etc...


![](https://hackmd.io/_uploads/S1_8Qgs0n.png)

#### Computational Graph
Computational graphs are a type of graph that can be used to represent mathematical expressions. 
These can be used for two different type of calculations:
1. Forward Computation 
2. Backward Computation

Let's take an example of a simple mathematical expression:
![](https://hackmd.io/_uploads/Hk_PTbsAh.png)

d=a+b
e=b-c
Y=d*e
we can draw a computational graph of the above equation as given in the above figure.

There are three computer graph equations 
1- addition of a and b giving output d.
2- subtraction of b and c giving output e.
3- Multiplication of d and e giving output Y.

#### Computational graph and Backpropogation 
Computational Graph and Backpropogation, both are importantcore concepts in deep learning for training neural network. 
__Forward Pass__: 
It is the procedure for evaluating the value of the mathematical expression represented by computational graph. Doing forward pass means we are passing the value from variables in forward direction from the left(input) to the right where the output is.

Let us consider an example,
Let us consider an example by giving some value to all of the inputs. Suppose, the following values are given to all of the inputs.

x=1,y=3,z=−3 By giving these values to the inputs, we can perform forward pass and get the following values for the outputs on each node.

First, we use the value of x = 1 and y = 3, to get p = 4.![](https://hackmd.io/_uploads/Sy7KKGo0h.png)


Forward Pass, then we use p = 4 and z = -3 to get g = -12. We go from left to right, forwards. image
![](https://hackmd.io/_uploads/BkhFtfiR3.png)

__Backward Pass:__ 
In the backward pass, our intention is to compute the gradients for each input with respect to the final output. These gradients are essential for training the neural network using gradient descent.

For example, we desire the following gradients.

Desired gradients ∂x∂f,∂y∂f,∂z∂f Backward pass (backpropagation) We start the backward pass by finding the derivative of the final output with respect to the final output (itself). Thus, it will result in the identity derivation and the value is equal to one.

∂g∂g=1 Our computational graph now looks as shown below 
![](https://hackmd.io/_uploads/rkZqTGiR2.png)


The main reason for doing this backwards is that when we had to calculate the gradient at x, we only used already computed values, and dq/dx (derivative of node output with respect to the same node's input). 

#### Vectorization
The term Vecorization describes the use of optimized, pre-compiled code written in a low-level language (eg: C language) to perform mathematical operations over a sequence of data. 
Vectorization allows the elimination of the for-loops in python code. This important in deep learning as we deal with large and many data sets in it.

In logical Regression,
![](https://hackmd.io/_uploads/rJ55MmiAn.png)
where x and w are both matrices with large vectors where x, w ϵ ℛ

So to compute the Z, we will need to write a for-loop for a non-vectorized function such as

    c= 0
    tic = time.time()
    for i in range(1000000):
      c +=w[i]*x[i]

For the vectorized version in NumPy, we simply calculate the dot product of the two vectors.

    c = np.dot(w,x)
    toc = time.time()

If we use the time function to calculate the time for both operations, we can see that the vectorized version is significantly faster than the non-vectorized version. The time factor especially gets compounded when we are dealing with very large matrixes.

    import time
    w = np.random.rand(1000000)
    x = np.random.rand(1000000)
    tic = time.time()
    c= np.dot(w,x)
    toc = time.time()
    print(c)
    print('Vectorized version: '+ str(1000*(toc-    tic))+'ms')
    c= 0
    tic = time.time()
    for i in range(1000000):
    c +=w[i]*x[i]
    toc = time.time()
    print(c)
    print("For loop: ",str(1000*(toc-tic))+'ms')
    250069.16704749785 
    Vectorized version: 2.0003318786621094ms 
    250069.16704750186 
    For loop:  236.63949966430664ms
    
As we can see from the code above, even though we have the same answers for the multiplication (250069.16), our dot product is almost 118x faster.
we can run this function on SIMB, the algorithm takes advantage of the parallelism which uses the native dot product to get the results faster.
![](https://hackmd.io/_uploads/BJEamQoCn.png)

#### Short Instruction/ Multiple Data
SIMD is short for Single Instruction/Multiple Data, while the term SIMD operations refer to a computing method that enables the processing of multiple data with a single instruction. In contrast, the conventional sequential approach using one instruction to process each individual data is called scalar operations. Using a simple summation as an example, the difference between the scalar and SIMD operations is illustrated below. See below for how each method handles the same four sets of additions.

![](https://hackmd.io/_uploads/SJm07XjR2.png)
In other words, this allows us to take advantage of the GPUs which are better at SIMD in the training algorithms faster than the CPUs.

#### Vectorizing Logistic Regression
We can vectorize the implementation of logistic regression, so they can process an entire training set, that is implement a single elevation of gradient descent with respect to an entire training set without using even a single explixit for a loop.

Let us first examine the four propogation steps of logistic regression. So let's say we have M training examples, then to make a prediction on the first example, we need to compute z.

    z(1)=wTx(1)+b
    a(1)=σ(z(1))
    z(2)=wTx(2)+b
    a(2)=σ(z(2))
    z(3)=wTx(3)+b
    a(3)=σ(z(3))

The above code tells us how to compute 'z' and then the activation for three examples.
This is to be done M times, because we have M traing examples.



#### __Vectorizing Logistic Regression's Gradient Output__
![](https://hackmd.io/_uploads/B1q6dXiRn.png)

#### Broadcasting in Python
The term broadcasting refers to how numpy treats arrays with different Dimension during arithmetic operations which lead to certain constraints, the smaller array is broadcast across the larger array so that they have compatible shapes. 

First we need to define a matrix capital X to be our training inputs, stacked together in different columns like
![](https://hackmd.io/_uploads/ry8r_Xi0h.png)

Using this matrix we can compute Z=Z(1)Z(2)Z(3)...Z(n) with one line of code. The matrix Z is a 1 by M matrix that's really a row vector.


Using matrix multiplication Z=WTX+b can be computed to obtain Z in one step.


    The python command is : z = np.dot(w.T, X) + B


Here W = (NX by 1), X is (NX by M) and b is (1 by M), multiplication and addition gives Z (1 by M).


Now there is a subtlety in Python , which is at here B is a real number or if you want to say you know 1x1 matrix, is just a normal real number.


But, when you add this vector to this real number, Python automatically takes this real number B and expands it out to this 1XM row vector.


This is called __broadcasting in Python__.








