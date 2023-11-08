
import numpy as np
import cupy as cp
import utils as us
from numpy import indices
from utils import _release_memory

class Dropout():
    def __init__(self, dropout_rate=0.1, data_type=np.float32):
        self.dropout_rate = dropout_rate
        self.data_type = data_type

    def forward(self, x, training=True):
        self.dropout_mask = np.random.binomial(1, 1-self.dropout_rate, x.shape).astype(self.data_type) if training else 1.0
        return x * self.dropout_mask

    def backward(self, grad_y):
        return grad_y * self.dropout_mask
    



class Embedding():
    def __init__(self, num_embeddings, embedding_dim, optimizer, data_type=cp.float32):
        self.layer_name = "embedding"
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.optimizer = optimizer
        self.data_type = data_type
        self.weights = None
        self.grad_weights = None
        self.init_weights()
        self.zero_grad()
        self.register()

    def init_weights(self):
        self.weights = cp.random.normal(0, self.num_embeddings ** -0.5, (self.num_embeddings, self.embedding_dim)).astype(self.data_type)

    def zero_grad(self):
        self.grad_weights = cp.zeros_like(self.weights)

    def register(self):
        weights_registered_name = '{}_{}'.format(self.layer_name, 'weights')
        cnt= self.optimizer.count_layers(weights_registered_name)
        self.weights_registered_name = "{}_{}".format(weights_registered_name, cnt)
        self.optimizer.register_params(self.weights_registered_name, self.weights)

    def forward(self, indices):
        self.indices = indices
        self.output = cp.take(self.weights, self.indices, axis=0)
        return self.output
    
    def backward(self, grad_y):
        self.grad_weights[self.indices] += grad_y
        return None

    def release_memory(self):
        del self.indices, self.output

    def update_weights(self):
        self.weights = self.optimizer.update(self.weights, self.grad_weights, self.weights_registered_name)
        self.release_memory()
        self.zero_grad()
        return
    


class LayerNormalization():
    def __init__(self, optimizer, normalized_shape, eps=1e-05, data_type=cp.float32):
        self.layer_name = "layernorm"
        self.optimizer = optimizer
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.data_type = data_type
        self.gamma = None
        self.beta = None
        self.mean = None
        self.var = None
        self.x_hat = None
        self.grad_gamma = None
        self.grad_beta = None
        self.init_weights()
        self.zero_grad()
        self.register()

    def init_weights(self):
        self.gamma = cp.ones((1, 1,self.normalized_shape), dtype=self.data_type)
        self.beta = cp.zeros((1, 1, self.normalized_shape), dtype=self.data_type)

    def zero_grad(self):
        self.grad_gamma = cp.zeros_like(self.gamma)
        self.grad_beta = cp.zeros_like(self.beta)

    def register(self):
        #This method gives a unique name to each layer and if layers occur more than once occur then adds a count in front of it
        self.layer_id = self.optimizer.count_layers(self.layer_name) 
        self.register_name = "{}_{}".format(self.layer_name, self.layer_id)
        self.optimizer.register_params("{}.gamma".format(self.register_name), self.gamma)
        self.optimizer.register_params("{}.beta".format(self.register_name), self.beta)

    def forward(self, x):
        self.x = x
        self.mean = cp.mean(x, axis=-1, keepdims=True)
        self.var = cp.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / cp.sqrt(self.var + self.eps)
        y = self.gamma * self.x_hat + self.beta
        return y
    
    def backward(self, grad):
        self.grad_gamma = cp.sum(grad * self.x_hat, axis=(0, 1), keepdims=True)
        self.grad_beta = cp.sum(grad, axis=(0, 1), keepdims=True)

        # Calculate gradient of x_hat
        dx_hat = grad * self.gamma

        # Calculate gradients of mean and var
        D = self.x.shape[-1]
        dvar = cp.sum(dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = cp.sum(dx_hat * -1 / cp.sqrt(self.var + self.eps), axis=-1, keepdims=True)

        # Calculate gradient of x
        dx_hat = dx_hat / cp.sqrt(self.var + self.eps)
        dx = dx_hat + (dvar * 2 * (self.x - self.mean) / D) + (dmean / D)

        return dx
    def release_memory(self):
        del self.grad_gamma, self.grad_beta

    def update_weights(self):
        self.gamma = self.optimizer.update(self.gamma, self.grad_gamma, "{}.gamma".format(self.register_name))
        self.beta = self.optimizer.update(self.beta, self.grad_beta, "{}.beta".format(self.register_name))
        
        
        
        
class Linear():
    def __init__(self, in_features, out_features, optimizer, use_bias=True, data_type=cp.float32):
        self.layer_name = "linear"
        self.in_features = in_features
        self.out_features = out_features
        self.optimizer = optimizer
        self.use_bias = use_bias
        self.weights = None
        self.bias= None
        self.grad_weights = None
        self.grad_bias = None
        self.data_type = data_type
        self.init_weights()
        self.zero_grad()
        self.register()


    def init_weights(self):
        sqrt_k = 1. / cp.sqrt(self.in_features)
        self.weights = cp.random.uniform(-sqrt_k, sqrt_k, (self.in_features, self.out_features)).astype(self.data_type)
        if self.use_bias:
            self.bias = cp.random.uniform(-sqrt_k, sqrt_k, self.out_features).astype(self.data_type)

    def zero_grad(self):
        self.grad_weights = cp.zeros_like(self.weights)
        if self.use_bias:
            self.grad_bias = cp.zeros_like(self.bias)
    
    def register(self):
        weights_registered_name = '{}_{}'.format(self.layer_name, 'weights')
        cnt= self.optimizer.count_layers(weights_registered_name)
        self.weights_registered_name = "{}_{}".format(weights_registered_name, cnt)
        self.optimizer.register_params(self.weights_registered_name, self.weights)
        if self.use_bias:
            bias_registered_name = '{}_{}'.format(self.layer_name, 'bias')
            cnt= self.optimizer.count_layers(bias_registered_name)
            self.bias_registered_name = "{}_{}".format(bias_registered_name, cnt)
            self.optimizer.register_params(self.bias_registered_name, self.bias)

    def forward(self, x):
        self.x = x
        self.output = x @ self.weights
        if self.use_bias:
            self.output += self.bias
        return self.output
    
    def backward(self, grad):
        self.grad_weights += cp.sum(cp.matmul(self.x.transpose(0, 2, 1), grad), axis=0)
        if self.use_bias:
            self.grad_bias += cp.sum(grad, axis=tuple(range(grad.ndim - 1)))
        self.grad = grad @ self.weights.T
        return self.grad
    
    def release_memory(self):
        del self.x, self.output
       

    def update_weights(self):
        self.weights = self.optimizer.update(self.weights, self.grad_weights, self.weights_registered_name)
        if self.use_bias:
            self.bias = self.optimizer.update(self.bias, self.grad_bias, self.bias_registered_name)
        self.release_memory()
        self.zero_grad()
        
        
        
        
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