import cupy as cp

class Softmax():
    def __init__(self):
        self.axis = -1

    def forward(self, x):
        e_x = cp.exp(x - cp.max(x, axis = self.axis, keepdims=True))
        self.y =  e_x / cp.sum(e_x, axis = self.axis, keepdims=True)
        del e_x
        return cp.nan_to_num(self.y, nan=0.)

    def backward(self, grad_y):
        # https://sgugger.github.io/a-simple-neural-net-in-numpy.html
        grad_x = self.y * (grad_y - (grad_y * self.y).sum(axis=self.axis, keepdims=True))
        del self.y
        return cp.nan_to_num(grad_x, nan=0.)

