
import numpy as np
class Softmax():
    def __init__(self):
        self.axis=-1 #We set axis to be the last axis as it contains the class probabilities for different classes
    
    def forward(self,x):
        
        #Subtracting the maximum x in the layer so as to prevent overflow or numerical instability
        e_x=np.exp(x-np.max(x,axis=self.axis,keepdims=True))
        
        #Assigns the probability 
        self.y=e_x/np.sum(e_x,axis=self.axis,keepdims=True)
        
        #Ensures potential nan values converted to 0
        return np.nan_to_num(self.y,nan=0.0) 
    
    def backward(self,grad_y):
        
        #grad_y is the gradient of loss wrt output of Softmax Layer 
        #grad_x is the gradient of loss wrt input of Softmax Layer

        grad_x=self.y *(grad_y-(self.y*grad_y).sum(axis=self.axis,keepdims=True))
       
        
        #Ensures potential nan values converted to 0
        return np.nan_to_num(grad_x,nan=0.0)




