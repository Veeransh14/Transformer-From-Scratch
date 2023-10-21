
import numpy as np
class LayerNormalization():
    def __init__(self, optimizer, normalized_shape, eps=1e-05, data_type=np.float32):
        self.layer_name = "layernorm" #Assigning the layer name as layer norm
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
        #gamma initialised to an array of ones same as that of input shape
        self.gamma = np.ones(self.normalized_shape).astype(self.data_type) 
        
        #beta initialised to an array of zeros same as that of input shape
        self.beta = np.zeros(self.normalized_shape).astype(self.data_type)

    #The zero_grad method initializes the gradient values for gamma and beta to arrays of zeros
    def zero_grad(self):
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

    def register(self):
        #This method gives a unique name to each layer and if layers occur more than once occur then adds a count in front of it
        self.layer_id = self.optimizer.count_layers(self.layer_name) 
        self.register_name = "{}_{}".format(self.layer_name, self.layer_id)
        self.optimizer.register_params("{}.gamma".format(self.register_name), self.gamma)
        self.optimizer.register_params("{}.beta".format(self.register_name), self.beta)

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        y = self.gamma * self.x_hat + self.beta
        return y

    def backward(self, grad):
        _, D = grad.shape

        # Calculate dL/dx_hat, dL/dvar, dL/dmean
        dx_hat = grad * self.gamma
        dvar = np.sum(dx_hat * (self.x_hat - self.mean) * (-0.5) * np.power(self.var + self.eps, -1.5), axis=1, keepdims=True)
        dmean = np.sum(dx_hat * (-1 / np.sqrt(self.var + self.eps)), axis=1, keepdims=True) + dvar * np.mean(-2 * (self.x_hat - self.mean), axis=1, keepdims=True)

        # Calculate dL/dx
        dx = dx_hat * (1 / np.sqrt(self.var + self.eps)) + dvar * (2 * (self.x_hat - self.mean) / D) + dmean / D

        # Calculate dL/dgamma, dL/dbeta
        self.grad_gamma += np.sum(grad * self.x_hat, axis=0, keepdims=True)
        self.grad_beta += np.sum(grad, axis=0, keepdims=True)

        return dx

    def release_memory(self):
        #Free up the memory by deleting unwanted variables after one step of forward and backward prop
        del self.grad_gamma, self.grad_beta

    def update_weights(self):
        self.gamma = self.optimizer.update(self.gamma, self.grad_gamma, "{}.gamma".format(self.register_name))
        self.beta = self.optimizer.update(self.beta, self.grad_beta, "{}.beta".format(self.register_name))
        self.zero_grad()






