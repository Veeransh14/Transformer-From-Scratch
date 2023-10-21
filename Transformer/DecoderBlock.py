import numpy as np
from Dropout import Dropout
from LayerNorm import LayerNormalization
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward
from RELU import RELU

class DecoderBlock:
    def __init__(self, optimizer, d_model=512, d_ff=2048, num_attention_heads=8, dropout_rate=0.1, data_type=np.float32):
        self.optimizer = optimizer
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.data_type = data_type
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
        self.dropout3 = Dropout(self.dropout_rate)
        self.layernorm1 = LayerNormalization(self.optimizer, self.d_model, 1e-5, self.data_type)
        self.layernorm2 = LayerNormalization(self.optimizer, self.d_model, 1e-5, self.data_type)
        self.layernorm3 = LayerNormalization(self.optimizer, self.d_model, 1e-5, self.data_type)
        self.multi_head_att1 = MultiHeadAttention(self.optimizer, self.d_model, self.num_attention_heads, self.dropout_rate, self.data_type)
        self.multi_head_att2 = MultiHeadAttention(self.optimizer, self.d_model, self.num_attention_heads, self.dropout_rate, self.data_type)
        self.ffn = PositionWiseFeedForward(self.optimizer, self.d_model, self.d_ff, self.dropout_rate, self.data_type)

    def forward(self, target, source, target_mask, src_tgt_mask, training=True):
        # Self-attention on the target sequence
        q1, k1, v1 = target, target, target
        attention_output1 = self.multi_head_att1.forward(q1, k1, v1, target_mask, training)
        attention_output1 = self.dropout1.forward(attention_output1, training)
        target_ = self.layernorm1.forward(target + attention_output1)

        # Cross-attention on the source sequence
        q2, k2, v2 = target_, source, source
        attention_output2 = self.multi_head_att2.forward(q2, k2, v2, src_tgt_mask, training)
        attention_output2 = self.dropout2.forward(attention_output2, training)
        x = self.layernorm2.forward(target_ + attention_output2)

        # Position-wise feedforward
        x_ = self.ffn.forward(x, training)
        x_ = self.dropout3.forward(x_, training)
        x = self.layernorm3.forward(x + x_)
        
        return x

    def backward(self, grad):
        # Backward pass for position-wise feedforward
        grad_res1 = self.layernorm3.backward(grad)
        grad = self.dropout3.backward(grad_res1)
        grad = self.ffn.backward(grad)
        grad = grad_res1 + grad

        # Backward pass for cross-attention
        grad_res2 = self.layernorm2.backward(grad)
        grad = self.dropout2.backward(grad_res2)
        grad, _ = self.multi_head_att2.backward(grad)
        grad = grad_res2 + grad

        # Backward pass for self-attention
        grad_res3 = self.layernorm1.backward(grad)
        grad = self.dropout1.backward(grad_res3)
        grad, _ = self.multi_head_att1.backward(grad)
        
        return grad_res3 + grad

    def update_weights(self):
        self.layernorm3.update_weights()
        self.ffn.update_weights()
        self.layernorm2.update_weights()
        self.multi_head_att2.update_weights()
        self.layernorm1.update_weights()
        self.multi_head_att1.update_weights()
