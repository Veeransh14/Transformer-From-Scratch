import cupy as cp
from numpy import indices
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
