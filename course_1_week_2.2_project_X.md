# Transformer From Scratch 


# Course 1-
# Week 2-

# Vecotrization
As the data increases or the complexity of a model increases, the machine learning algorithms tend to get slower, and generating predictions can take many hours or days, then we have to rely on computational resources (like GPUs or more CPUs) to reduce the waiting time.

Vectorization is the process of reducing the algorithm’s running time without using external computational resources.

The idea of Vectorization is to remove explicit ‘for loops’ from the code by replacing it with a vectorized implementation that yields the same output.

# Vectorization in Machine Learning
In Machine Learning or Deep Learning, we often train our models on large datasets. So it becomes crucial that our code is time optimized, else the model will take several hours or days to yield results.

Let us take the example of the Logistic Regression ML algorithm,

We have the below equation,

# Vectorization in Machine Learning
In Machine Learning or Deep Learning, we often train our models on large datasets. So it becomes crucial that our code is time optimized, else the model will take several hours or days to yield results.

Let us take the example of the Logistic Regression ML algorithm,

We have the below equation,

 ![Alt text](image-16.png)

The first step in calculating the value of Z is to multiply the two vectors W and X. We can multiply the two vectors in two ways vectorized and non-vectorized.

# Implementing Vectorization in Python with numpy library
   # Importing Libraries
    import numpy as np    # Using numpy for doing operations on the arrays
    import time    # Using time to determine the time taken by different operations

   # Vectorized demonstration
    CODE-

    before_time = time.time()
    Z = np.dot(W.transpose(), X)
    after_time = time.time()
    print("Time taken by the dot product is: ", (after_time - before_time)*1000, " ms")
    print(round(Z, 5))

 In short vectoriztion help us reduce th time complexity of the code because it provides us alternative past against loops hence the order O of time complexity in reduced to great extent.

# Vectorizing Logistic Regression


You can vectorize the implementation of logistic regression, so they can process an entire training set, that is implement a single elevation of grading descent with respect to an entire training set without using even a single explicit for loop.

Let's first examine the four propagation steps of logistic regression. So, if you have M training examples, then to make a prediction on the first example, you need to compute Z.

z(1)=wTx(1)+b
a(1)=σ(z(1))
z(2)=wTx(2)+b
a(2)=σ(z(2))
z(3)=wTx(3)+b
a(3)=σ(z(3))

The above code tells how to compute Z and then the activation for three examples.
And you might need to do this M times, if you have M training examples.
So in order to carry out the four propagation step, that is to compute these predictions on our M training examples, there is a way to do so, without needing an explicit for loop.
First , we defined a matrix capital X to be our training inputs, stacked together in different columns.
training inputs, stacked together in different columns
This is a (NX by M) matrix.
Using this matrix we can compute Z=Z(1)Z(2)Z(3)...Z(n) with one line of code. The matrix Z is a 1 by M matrix that's really a row vector.
Using matrix multiplication Z=WTX+b can be computed to obtain Z in one step.
The python command is : z = np.dot(w.T, X) + B
Here W = (NX by 1), X is (NX by M) and b is (1 by M), multiplication and addition gives Z (1 by M).
Now there is a subtlety in Python , which is at here B is a real number or if you want to say you know 1x1 matrix, is just a normal real number.
But, when you add this vector to this real number, Python automatically takes this real number B and expands it out to this 1XM row vector.
This is called broadcasting in Python.




# Vectorizing Logistic Regression's Gradient Output
We can use vectorization to also perform the gradient computations for all M training samples. To derive a very efficient implementation of logistic regression.
For the gradient computation , the first step is to compute dz(1) for the first example, which could be a(1)−y(1) and then dz(2)=a(2)−y(2) and so on. And so on for all M training examples.
Lets define a new variable , dZ is going to be dz(1),dz(2),...,dz(m). Again, all the D lowercase z variables stacked horizontally. So, this would be 1 by m matrix or alternatively a 'm' dimensional row vector.
Previously , we'd already figured out how to compute A and Y as shown below and you can see for yourself that dz can be computed as just A minus Y because it's going to be equal to a1 - y1, a2 - y2, and so on.
So, with just one line of code , you can compute all of this at the same time. By a simple matrix multiplication as shown below.
logistic-regression-non-vectorized
Now after dz we compute dw1, dw2 and db as shown in the below figure.
To compute dw , we initialize dw to zero to a vector of zeroes. Then we still have to loop over examples where we have dw + = x1 * dz1, for the first training example and so on.
logistic-regression-non-vectorized
The above code line can vectorize the entire computation of dw in one line.
Similarly for the vectorize implementation of db was doing is basically summing up, all of these dzs and then dividing by m. This can be done using just one line in python as: db = 1/m * np.sum(dz)
And so the gradient descent update then would be you know W gets updated as w minus the learning rate times dw which was just computed above and B is update as B minus the learning rate times db.
Now, I know I said that we should get rid of explicit full loops whenever you can but if you want to implement multiple adjuration as a gradient descent then you still need a full loop over the number of iterations. So, if you want to have a thousand deliberations of gradient descent, you might still need a full loop over the iteration number.
There is an outermost full loop like that then I don't think there is any way to get rid of that full loop.
A technique in python which makes our code easy for implementation is called broadcasting

# Broadcasting In Python
The term broadcasting refers to how numpy treats arrays with different Dimension during arithmetic operations which lead to certain constraints, the smaller array is broadcast across the larger array so that they have compatible shapes. 
Broadcasting provides a means of vectorizing array operations so that looping occurs. It does this without making needless copies of data and which leads to efficient algorithm implementations. There are cases where broadcasting is a bad idea because it leads to inefficient use of memory that slow down the computation.
We use word array and matrix as to convey the same meaning here.
![Alt text](image-17.png)
