import cupy as cp
class Dropout():
    def __init__(self, dropout_rate=0.1, data_type=cp.float32):
        self.dropout_rate = dropout_rate
        self.data_type = data_type

    def forward(self, x, training=True):
        self.dropout_mask = cp.random.binomial(1, 1-self.dropout_rate, x.shape).astype(self.data_type) if training else 1.0
        return x * self.dropout_mask

    def backward(self, grad_y):
        return grad_y * self.dropout_mask