import numpy as np

class Embedding():
    #The constructor initializes several key attributes for the Embedding class
    def __init__(self, num_embeddings, embedding_dim, optimizer=None, data_type=np.float32):
        self.layer_name = "embedding" #Assign a name to layer 
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
        
        #To ensure the weights are not too random we set limits as [-sqrt_k,sqrt_k] so that it is centered around zero
        sqrt_k = 1. / np.sqrt(self.embedding_dim)
        self.weights = np.random.uniform(-sqrt_k, sqrt_k, (self.num_embeddings, self.embedding_dim)).astype(self.data_type)

    def zero_grad(self):
        #grad_weights will accumulate gradients during back prop 
        #we ensure that it has same size as weights so that optimization can be done 
        self.grad_weights = np.zeros_like(self.weights)
        
    #The register method handles the registration of the embedding layer with an optimizer 
    def register(self):
        if self.optimizer is not None:
            #generates a unique name for the embedding weights.
            #The name is a combination of the layer_name and 'weights' to ensure uniqueness.
            weights_registered_name = '{}_{}'.format(self.layer_name, 'weights')
            
            #A counter is used to distinguish between different layers with the same name
            cnt = self.optimizer.count_layers(weights_registered_name)
            self.weights_registered_name = "{}_{}".format(weights_registered_name, cnt)
            
            #register the embedding weights with the optimizer using the generated name.
            self.optimizer.register_params(self.weights_registered_name, self.weights)
    
    #forward maps input tokens to their corresponding embeddings
    def forward(self, x):
        #Here x represents the indices of words 
        self.x = x
        
        #We map out the corresponding embedding weights stored in self.weights using x
        return np.take(self.weights, x, axis=0)
    
    
    def backward(self, grad):
        self.grad_weights[:] = 0
        
        #Update the gradients
        for i, x in enumerate(self.x):
            self.grad_weights[x] += grad[i]
        if self.optimizer is not None:
            self.optimizer.update(self.weights_registered_name, self.grad_weights)
        return grad

    def update_weights(self):
        self.weights = self.optimizer.update(self.weights_registered_name, self.weights)

    def release_memory(self):
        del self.x


