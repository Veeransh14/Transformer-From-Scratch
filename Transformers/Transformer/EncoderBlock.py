

import numpy as np
from Dropout import Dropout
from LayerNorm import LayerNormalization
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward
class EncoderBlock():
    def __init__(self, d_model, d_ff, optimizer, num_attention_heads, dropout_rate, data_type):
        self.d_model = d_model
        self.d_ff = d_ff
        self.optimizer = optimizer
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.data_type = data_type
        
        #We are initializing the instance of the various classes 
        self.dropout1 = Dropout(self.dropout_rate, self.data_type)
        self.dropout2 = Dropout(self.dropout_rate, self.data_type)
        self.layernorm1 = LayerNormalization(self.optimizer, normalized_shape=self.d_model, eps=1e-5, data_type=self.data_type)
        self.layernorm2 = LayerNormalization(self.optimizer, normalized_shape=self.d_model, eps=1e-5, data_type=self.data_type)
        self.multi_head_attention = MultiHeadAttention(self.optimizer, self.d_model, self.num_attention_heads, self.dropout_rate, self.data_type)
        self.ffn = PositionWiseFeedForward(self.optimizer, self.d_model, self.d_ff, self.dropout_rate, self.data_type)

    def forward(self, x, mask, training=True):
        #Sharing the same tensors across k,q,v improves efficiency and helps us to learn context dependent relationship 
        #We can think k,q,v are the linear projections of tensor x
        #q is the current position
        #k represents the other positions 
        #v represents the value stored at each position
        q, k, v = x, x, x
        attention_output = self.multi_head_attention.forward(q, k, v, mask, training)
        attention_output = self.dropout1.forward(attention_output, training)
        x = self.layernorm1.forward(x + attention_output)#Residual connection
        ffn_output = self.ffn.forward(x, training)
        ffn_output  = self.dropout2.forward(ffn_output, training)
        output = self.layernorm2.forward(x + ffn_output)#Residual connection
        return output

    def backward(self, grad):
        grad_res = self.layernorm2.backward(grad) #The gradient is backpropagated through the second layer normalization
        grad = self.dropout2.backward(grad_res)#The gradient is backpropagated through dropout applied to feedforward network output
        grad = self.ffn.backward(grad)# The gradient is further backpropagated through the feed-forward network
        grad = grad_res + grad #The gradients are added together, effectively accumulating gradients.
        grad_res = self.layernorm1.backward(grad)#The gradient is backpropagated through the first layer normalization 
        grad = self.dropout1.backward(grad_res)#The gradient is backpropagated through the dropout applied to the attention output
        grad_q, grad_k, grad_v = self.multi_head_attention.backward(grad)#The gradient is further backpropagated through the multi-head self-attention mechanism
        grad = grad_res + grad_q + grad_k + grad_v #The gradients are added together
        return grad

    def update_weights(self):
        self.layernorm2.update_weights()
        self.ffn.update_weights()
        self.layernorm1.update_weights()
        self.multi_head_attention.update_weights()


# In[ ]:




