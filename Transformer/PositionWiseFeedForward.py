
import numpy as np
from Linear import linear
from Dropout import Dropout
from RELU import RELU
class PositionWiseFeedForward():
    #d_ff is dimensionality of the hidden layer
    def __init__(self, optimizer, d_model, d_ff, dropout_rate=0.1, data_type=np.float32):
        self.fc1 = linear(d_model, d_ff, optimizer, use_bias=True, data_type=data_type)
        self.fc2 = linear(d_ff, d_model, optimizer, use_bias=True, data_type=data_type)
        self.relu = RELU()
        self.dropout = Dropout(dropout_rate, data_type)

    def forward(self, x, training=True):
        #Dimensionality Expansion: The input at each position in the sequence is projected into a higher-dimensional space using the first linear layer. This step allows the model to capture more complex patterns in the data.
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.dropout.forward(x, training)
        
        #Dimensionality Reduction: The output of the activation function is then projected back into the original lower-dimensional space using the second linear layer. This step condenses the information learned during the dimensionality expansion and mapping, ensuring that the output remains in the same dimension as the input.
        x = self.fc2.forward(x)
        
        return x

    def backward(self, grad):
        
        #We get gradient of loss wrt the output of layer as parameter and we return gradient of loss wrt input of the layer 
        grad = self.fc2.backward(grad)
        grad = self.dropout.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)
        return grad

    def update_weights(self):
        self.fc1.update_weights()
        self.fc2.update_weights()






