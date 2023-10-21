
import numpy as np
class PositionalEncoding():
    def __init__(self,max_len,d_model,data_type=np.float32):
        self.data_type = data_type
        self.pe = self.positional_encoding(max_len, d_model) #self.pe holds the positional encoding matrix
    
    def positional_encoding(self,max_len,d_model):
        position=np.arange(max_len)[:,np.newaxis] #Column vector with values from 0 to max_len-1
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((1, max_len, d_model))
        pe[0, :, 0::2] = np.sin(position * div_term) # For Even terms
        pe[0, :, 1::2] = np.cos(position * div_term) # For odd terms 
        return pe.astype(self.data_type)
    
    def forward(self,x):
        batch_size, sequence_length, d_model = x.shape
        #The np.tile function creates a new array by repeating the self.pe array multiple times in the batch dimension
        expanded_pe = np.tile(self.pe, (batch_size, 1, 1))  
        return x + expanded_pe[:, :sequence_length, :]  #Broadcasting
    
    def backward(self, grad):
        
        #Gradient of loss wrt input or output of the layer remains the same 
        
        return grad

        




