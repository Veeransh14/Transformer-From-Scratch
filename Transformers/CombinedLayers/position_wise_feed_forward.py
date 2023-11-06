import cupy as cp
from relu import ReLU
from linear import Linear
from dropout import Dropout
class PositionWiseFeedForward():
    def __init__(self, optimizer, d_model, d_ff, dropout_rate=0.1, data_type=cp.float32):
        self.fc1 = Linear(d_model, d_ff, optimizer, use_bias=True, data_type=data_type)
        self.fc2 = Linear(d_ff, d_model, optimizer, use_bias=True, data_type=data_type)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate, data_type)

    def forward(self, x, training=True):
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.dropout.forward(x, training)
        x = self.fc2.forward(x)
        return x

    def backward(self, grad):
        grad = self.fc2.backward(grad)
        grad = self.dropout.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)
        return grad

    def update_weights(self):
        self.fc1.update_weights()
        self.fc2.update_weights()
