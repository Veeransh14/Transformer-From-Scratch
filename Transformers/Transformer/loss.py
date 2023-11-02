#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class CrossEntropy():
    def __init__(self, padding_id, vocab_size):
        self.eps = 1e-6
        self.padding_id = padding_id #padding_id is an identifier for padding tokens in the vocabulary.
        self.vocab_size = vocab_size#vocab_size is the size of the vocabulary (the number of unique words)

    #one_hot is a method that converts a label (index of the correct class) into a one-hot encoded vector.
    def one_hot(self, label):
        
        #label refers to the ground truth class 
        one_hot = np.zeros((len(label), self.vocab_size))
        one_hot[np.arange(len(label)), label] = 1
        self.one_hot_label = one_hot
        return one_hot

    def forward(self, pred, label):
        one_hot_label = self.one_hot(label) #converts the true label label into a one-hot encoded vector.
        loss = -(one_hot_label * np.log(pred + self.eps)).sum(axis=-1)
        loss[label == self.padding_id] = 0 #the loss is set to 0 for those cases where true prediction is made
        return loss

    def grad(self, pred, label):
        grad = pred - self.one_hot_label
        grad[label.reshape(-1, 1) == self.padding_id] = 0 #the grad is 0 when true prediction is made
        return grad
class MSE():
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self):
        loss = ((self.y_pred - self.y_true) ** 2).mean()
        return loss

    def grad(self):
        grad = 2 * (self.y_pred - self.y_true)
        return grad


# In[ ]:




