try:
    import cupy as cp
except ImportError:
    import numpy as cp
from embedding import Embedding
from dropout import Dropout
from positional_encoding import PositionalEncoding
from encoder_block import EncoderBlock
class Encoder():
    def __init__(self, optimizer, vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type):
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model, optimizer, data_type)
        self.dropout = Dropout(dropout_rate, data_type)
        self.positional_encoding = PositionalEncoding(max_len, d_model, data_type)
        self.encoder_layers = [EncoderBlock(d_model, d_ff, optimizer, num_attention_heads, dropout_rate, data_type) for _ in range(block_num)]

    def forward(self, x, mask, training):
        x = self.embedding.forward(x) * cp.sqrt(self.d_model)
        x = self.positional_encoding.forward(x)
        x = self.dropout.forward(x, training)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward(x, mask, training)
        return x

    def backward(self, grad):
        for encoder_layer in reversed(self.encoder_layers):
            grad = encoder_layer.backward(grad)
        grad = self.dropout.backward(grad)
        grad = self.positional_encoding.backward(grad) * cp.sqrt(self.d_model)
        grad = self.embedding.backward(grad)
        return grad

    def update_weights(self):
        for encoder_layer in reversed(self.encoder_layers):
            encoder_layer.update_weights()
        self.embedding.update_weights()
