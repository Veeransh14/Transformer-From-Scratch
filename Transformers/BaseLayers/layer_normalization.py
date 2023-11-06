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
