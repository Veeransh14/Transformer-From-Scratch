import cupy as cp
class Dropout():
    #Initialising dropout rate and data type
    def __init__(self, dropout_rate=0.1, data_type=cp.float32):
        self.dropout_rate = dropout_rate
        self.data_type = data_type

    #Dropout mask is set using random binomial distribution if using it for training  else 1.0 for inference 
    #Number of trails is 1 for each neuron 
    #Dropout_mask has same shape as x
    #Probability of success that is the mask value to be 1.0 is  1-self.dropout_rate.

    def forward(self, x, training=True):
        self.dropout_mask = cp.random.binomial(1, 1-self.dropout_rate, x.shape).astype(self.data_type) if training else 1.0
        return x * self.dropout_mask

    #Input parameter- grad is the derivative of the loss wrt output of this layer 
    #We mask out the ones we are not using so we have to also mask out their grads 
    #We return gradient of loss wrt input of the layer 
    def backward(self, grad_y):
        return grad_y * self.dropout_mask
