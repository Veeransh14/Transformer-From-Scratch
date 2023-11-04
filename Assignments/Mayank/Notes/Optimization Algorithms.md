# Optimization Algorithms 
Optimization algorithms:Optimization algorithms are a class of algorithms that are used to find the best possible solution to a given problem. The goal of an optimization algorithm is to find the optimal solution that minimizes or maximizes a given objective function.

## Table of Content 
- 1. Mini Batch Gradient-Descent
- 2. Exponentially Weighted Moving Average
- 3. Bias Correction in Exponentially Weighted Moving Average
- 4. Gradient Descent with Momentum
- 5. Adam Optimization Algorithm
- 6. Learning Rate Decay 
- 7. Problem of Local Optima 

#### Mini Batch Gradient-Descent
We have seen the Batch Gradient Descent. We have also seen the Stochastic Gradient Descent. Batch Gradient Descent can be used for smoother curves. SGD can be used when the dataset is large. Batch Gradient Descent converges directly to minima. SGD converges faster for larger datasets. But, since in SGD we use only one example at a time, we cannot implement the vectorized implementation on it. This can slow down the computations. To tackle this problem, a mixture of Batch Gradient Descent and SGD is used.

Neither we use all the dataset all at once nor we use the single example at a time. We use a batch of a fixed number of training examples which is less than the actual dataset and call it a mini-batch. Doing this helps us achieve the advantages of both the former variants we saw. So, after creating the mini-batches of fixed size, we do the following steps in one epoch:

- Pick a mini-batch
- Feed it to Neural Network
- Calculate the mean gradient of the mini-batch
- Use the mean gradient we calculated in step 3 to update the weights
- Repeat steps 1–4 for the mini-batches we created

![](https://hackmd.io/_uploads/r1N8ZrzkT.png)
So, when we are using the mini-batch gradient descent we are updating our parameters frequently as well as we can use vectorized implementation for faster computations.

#### Exponentially Weighted Moving Average 
The Exponentially Weighted Moving Average (EWMA) is commonly used as a smoothing technique in time series. However, due to several computational advantages (fast, low-memory cost), the EWMA is behind the scenes of many optimization algorithms in deep learning, including Gradient Descent with Momentum, RMSprop, Adam, etc.

In order to compute the EWMA, you must define one parameter β. This parameter decides how important the current observation is in the calculation of the EWMA.

![](https://hackmd.io/_uploads/SyDGGHzJT.png)

Lets make an example based on the temperatures of Paris, France, in 2019
![](https://hackmd.io/_uploads/rJN4MHf1a.png)
Define: 
![](https://hackmd.io/_uploads/S19EfHMkT.png)
For this example, suppose that β = 0.9, so the EWA aims to combine the temperature of the current day with the previous temperatures.

![](https://hackmd.io/_uploads/SkZIzBMJa.png)
In general to compute the EWA for a given weight parameter β we use

![](https://hackmd.io/_uploads/S1BDGBGy6.png)
If we plot this in red, we can see that what we get is a moving average of the daily temperature, it’s like a smooth, less noisy curve.
![](https://hackmd.io/_uploads/Syb5GHG1a.png)
To understand the meaning of the parameter β, you can think of the value

![](https://hackmd.io/_uploads/r1zsGrGJa.png)
as the numbers of observations used to adapt your EWA.
![](https://hackmd.io/_uploads/rJQ2GHG16.png)

#### Bias Correction in Exponentially Weighted Moving Average

Making EWMA more accurate — Since the curve starts from 0, there are not many values to average on in the initial days. Thus, the curve is lower than the correct value initially and then moves in line with expected values.

Figure: The ideal curve should be the GREEN one, but it starts as the PURPLE curve since the values initially are zero

![](https://hackmd.io/_uploads/SkINmSGk6.png)


Example: Starting from t=0 and moving forward,
Vsub0 = 0Vsub1 = 0.98Vsub 0+0.02θsub1 = 0.020θsub1
Vsub2 = 0.98Vsub1 + 0.02θsub2 = 0.0196θsub1+0.02θsub2

The initial values of Vt will be very low which need to be compensated.
Make Vt = Vt/1−βsupt
for t=2, 1−βsupt= 1−0.9⁸² = 0.0396 (Bias Correction Factor)

Vsub2 = V2/0.0396 = 0.0196θsub1 + 0.02θsub2 / 0.0396

When t is large, 1/1−βsupt =1, hence bias correction factor has no effect when t is sufficiently large.

#### Gradient Descent with Momentum 
The problem with gradient descent is that the weight update at a moment (t) is governed by the learning rate and gradient at that moment only. It doesn’t take into account the past steps taken while traversing the cost space.
![](https://hackmd.io/_uploads/H1lsXBzka.png)

It leads to the following problems.

- The gradient of the cost function at saddle points( plateau) is negligible or zero, which in turn leads to small or no weight updates. Hence, the network becomes stagnant, and learning stops
- The path followed by Gradient Descent is very jittery even when operating with mini-batch mode

![](https://hackmd.io/_uploads/BkZxEHzkT.png)

Let’s assume the initial weights of the network under consideration correspond to point A. With gradient descent, the Loss function decreases rapidly along the slope AB as the gradient along this slope is high. But as soon as it reaches point B the gradient becomes very low. The weight updates around B is very small. Even after many iterations, the cost moves very slowly before getting stuck at a point where the gradient eventually becomes zero.

In this case, ideally, cost should have moved to the global minima point C, but because the gradient disappears at point B, we are stuck with a sub-optimal solution.

#### Adam Optimization Algorithm
- Stands for Adaptive Moment Estimation.
- Adam optimization and RMSprop are among the optimization algorithms that worked very well with a lot of NN architectures.
- Adam optimization simply puts RMSprop and momentum together
- Hyperparameters for Adam:
    - Learning rate: needed to be tuned.
    - beta1: parameter of the momentum - 0.9 is recommended by default.
    - beta2: parameter of the RMSprop - 0.999 is recommended by default.
    - epsilon: 10^-8 is recommended by default.

#### Learning Rate Decay

- Slowly reduce learning rate.
- As mentioned before mini-batch gradient descent won’t reach the optimum point (converge). But by making the learning rate decay with iterations it will be much closer to it because the steps (and possible oscillations) near the optimum are smaller.
- One technique equations islearning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate_0
- epoch_num is over all data (not a single mini-batch).
- Other learning rate decay methods (continuous):
- learning_rate = (0.95 ^ epoch_num) * learning_rate_0
- learning_rate = (k / sqrt(epoch_num)) * learning_rate_0
- Some people perform learning rate decay discretely - repeatedly decrease after some number of epochs.
- Some people are making changes to the learning rate manually.
- decay_rate is another hyperparameter.

#### The Problem of Local Optima
- The normal local optima is not likely to appear in a deep neural network because data is usually high dimensional. For point to be a local optima it has to be a local optima for each of the dimensions which is highly unlikely.
- It’s unlikely to get stuck in a bad local optima in high dimensions, it is much more likely to get to the saddle point rather to the local optima, which is not a problem.
- Plateaus can make learning slow:
    - Plateau is a region where the derivative is close to zero for a long time.
    - This is where algorithms like momentum, RMSprop or Adam can help.
