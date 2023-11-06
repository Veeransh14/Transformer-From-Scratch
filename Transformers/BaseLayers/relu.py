import cupy as cp

class ReLU():        
    def forward(self, x):
        self.x = x
        return cp.maximum(0., x)

    def backward(self, grad):
        # grad = grad * cp.where(self.x <= 0, 0, 1).astype(self.x.dtype)
        grad[self.x <= 0] = 0
        # x can be deleted here because it is not used in backward pass and forward pass is already done
        del self.x
        return grad