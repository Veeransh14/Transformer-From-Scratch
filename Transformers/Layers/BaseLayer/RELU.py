
import numpy as np
class RELU():
    
    def forward(self,x):
        self.x=x
        return np.maximum(0,x)
    
    def backward(self,grad):#Input parameter grad is the gradient of loss wrt output of the layer
        
        grad[self.x<=0]=0 #We modify it to become gradient wrt input of the layer
        
        del self.x #We don't need self.x after one cycle of forward and back prop so we delete it to conserve memory
        
        return grad







