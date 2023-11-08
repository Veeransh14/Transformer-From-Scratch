
import imp
import cupy as cp

from embedding import Embedding
from dropout import Dropout
from linear import Linear
from softmax import Softmax
from positional_encoding import PositionalEncoding
from decoder_block import DecoderBlock
from layer_normalization import LayerNormalization
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward

class Decoder():
    def __init__(self, optimizer, vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type):
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model, optimizer, data_type)
        self.dropout = Dropout(dropout_rate, data_type)
        self.positional_encoding = PositionalEncoding(max_len, d_model, data_type)
        self.decoder_layers = [DecoderBlock(optimizer, d_model, d_ff, num_attention_heads, dropout_rate, data_type) for _ in range(block_num)]
        self.fc = Linear(d_model, vocab_size, optimizer, True, data_type)
        self.softmax = Softmax()

    def forward(self, target, source, target_mask, src_tgt_mask, training):
        target = self.embedding.forward(target) * cp.sqrt(self.d_model)
        target = self.positional_encoding.forward(target)
        target = self.dropout.forward(target, training)
        for decoder_layer in self.decoder_layers:
            target =  decoder_layer.forward(target, source, target_mask, src_tgt_mask, training)
        target = self.fc.forward(target)
        target = self.softmax.forward(target)
        return target

    def backward(self, grad):
        
     
        grad = self.softmax.backward(grad)
        
        grad = self.fc.backward(grad)
        
        self.grad_source_sum = 0
        for decoder_layer in reversed(self.decoder_layers):
            grad, grad_source = decoder_layer.backward(grad)
            self.grad_source_sum += grad_source
        grad = self.dropout.backward(grad)
        grad = self.positional_encoding.backward(grad) * cp.sqrt(self.d_model)
        grad = self.embedding.backward(grad)
        return grad

    def update_weights(self):
        self.fc.update_weights()
        for decoder_layer in reversed(self.decoder_layers):
            decoder_layer.update_weights()
        self.embedding.update_weights()
        
        
        

class DecoderBlock():
    def __init__(self, optimizer, d_model, d_ff, num_attention_heads, dropout_rate, data_type):
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

        q, k, v = target, target, target
        attention_output1 = self.multi_head_att1.forward(q, k, v, target_mask, training)
        attention_output1 = self.dropout1.forward(attention_output1, training)
        target_ = self.layernorm1.forward(target + attention_output1)
        q, k, v = target_, source, source
        attention_output2 = self.multi_head_att2.forward(q, k, v, src_tgt_mask, training)
        attention_output2 = self.dropout2.forward(attention_output2, training)
        x = self.layernorm2.forward(target_ + attention_output2)
        x_ = self.ffn.forward(x, training)
        x_ = self.dropout3.forward(x_, training)
        x = self.layernorm3.forward(x + x_)
        return x

    def backward(self, grad):
        grad_res1 = self.layernorm3.backward(grad)
        grad = self.dropout3.backward(grad_res1)
        grad = self.ffn.backward(grad)
        grad = grad_res1 + grad
        grad_res2 = self.layernorm2.backward(grad)
        grad = self.dropout2.backward(grad_res2)
        grad, grad_src_k , grad_src_v = self.multi_head_att2.backward(grad)
        grad = grad_res2 + grad
        grad_res3 = self.layernorm1.backward(grad)
        grad = self.dropout1.backward(grad_res3)
        grad_tgt_q, grad_tgt_k, grad_tgt_v = self.multi_head_att1.backward(grad)
        return grad_res3 + grad_tgt_q + grad_tgt_k + grad_tgt_v, grad_src_k + grad_src_v

    def update_weights(self):
        self.layernorm3.update_weights()
        self.ffn.update_weights()
        self.layernorm2.update_weights()
        self.multi_head_att2.update_weights()
        self.layernorm1.update_weights()
        self.multi_head_att1.update_weights()