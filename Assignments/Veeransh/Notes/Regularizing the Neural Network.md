# Regularizing the Neural Network
Neural networks can learn to represent complex relationships between network inputs and outputs. This representation power helps them to perform better than traditional machine learning algorithms in computer vision and natural language processing tasks.

## Table of content:
- 1. Regularization
- 2. Why does Regularization reduce Overfitting
- 3. Dropout
- 4. Understanding Dropout 
- 5. Other Regularization Methods 

#### Regularization
There are many challenges associated with training neural networks, one of them is overfitting. When a neural network overfits on the training dataset, it learns an overly complex representation that models the training dataset too well. As a result, it performs exceptionally well on the training dataset but generalizes poorly to unseen test data.

Regularization techniches help improve a neural network's generalization ability by reducing overfitting. They do this by minimizing needless complexity and exposing the network to more diverse data. 

#### Why does Regularization reduce Overfitting
Regularization is a technique that adds information to a model to prevent the occurrence of overfitting. It is a type of regression that minimizes the coefficient estimates to zero to reduce the capacity (size) of a model. In this context, the reduction of the capacity of a model involves the removal of extra weights.
Some common Regularization techniques are:
- Early Stopping 
- L1 and L2 regularization
- Data augmentation
- Addition of noise 
- Dropout 

#### Dropout
In traditional machine learning, model ensembling helps reduce overfitting and improve model performance. For a simple classification problem, we can take one of the following approaches:

Train multiple classifiers to solve the same task.

Train different instances of the same classifier for different subsets of the training dataset.


For a simple classification model, ensemble technique such as bagging involves training the same classifier—on different subsets of training data—sampled with replacement. Suppose there are N such instances. At test time, the test sample is run through each classifier, and an ensemble of their predictions is used.

In general, the performance of an ensemble is at least as good as the individual models; it cannot be worse than that of the individual models.

If we were to transpose this idea to neural networks, we could try doing the following (while identifying the limitations of this approach):

Train multiple neural networks with different architectures. Train a neural network on different subsets of the training data. However, training multiple neural networks is prohibitively expensive.

Even if we train N different neural networks, running the data point through each of the N models—at test time—introduces substantial computational overhead.


Dropout is a regularization technique that addresses both of the above concerns.

#### Understanding Dropout
Let’s consider a simple neural network:

![](https://hackmd.io/_uploads/ByFxdEzkp.png)
Dropout involves dropping neurons in the hidden layers and (optionally) the input layer. During training, each neuron is assigned a “dropout”probability, like 0.5.

With a dropout of 0.5, there’s a 50% chance of each neuron participating in training within each training batch. This results in a slightly different network architecture for each batch. It is equivalent to training different neural networks on different subsets of the training data.
![](https://hackmd.io/_uploads/HJcW_Nfy6.png)
The weight matrix is initialized once at the beginning of the training. In general, for the k-th batch, backpropagation occurs only along those paths only through the neurons present for that batch. Meaning only the weights corresponding to neurons that are present get updates.

At test time, all the neurons are present in the network. So how do we account for the dropout during training? We weight each neuron’s output by the same probability p – proportional to the fraction of time the neuron was present during the training.

#### Other Regularization Method
##### Data Augmentation
Data augmentation is a regularization technique that helps a neural network generalize better by exposing it to a more diverse set of training examples. As deep neural networks require a large training dataset, data augmentation is also helpful when we have insufficient data to train a neural network.

Let’s take the example of image data augmentation. Suppose we have a dataset with N training examples across C classes. We can apply certain transformations to these N images to construct a larger dataset.

![](https://hackmd.io/_uploads/SJVGFEf1a.png)
All in all, we can apply any label-invariant transformation to perform data augmentation [1]. The following are some examples:

Color space transformations such as change of pixel intensities

Rotation and mirroring

Noise injection, distortion, and blurring

##### Early Stopping 
Early stopping is one of the simplest and most intuitive regularization techniques. It involves stopping the training of the neural network at an earlier epoch; hence the name early stopping.

If the training error becomes too low and reaches arbitrarily close to zero, then the network is sure to overfit on the training dataset. Such a neural network is a high variance model that performs badly on test data that it has never seen before despite its near-perfect performance on the training samples.

Therefore, if we can prevent the training loss from becoming arbitrarily low, the model is less likely to overfit on the training dataset, and will generalize better.

So how do we do it in practice? You can monitor one of the following:

- The change in metrics such as validation error and validation accuracy
- The change in the weight vector

