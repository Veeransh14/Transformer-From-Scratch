
import numpy as np

from numpy import indices




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
    def __init__(self, num_embeddings, embedding_dim, optimizer, data_type=np.float32):
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
        self.weights = np.random.normal(0, self.num_embeddings ** -0.5, (self.num_embeddings, self.embedding_dim)).astype(self.data_type)

    def zero_grad(self):
        self.grad_weights = np.zeros_like(self.weights)

    def register(self):
        weights_registered_name = '{}_{}'.format(self.layer_name, 'weights')
        cnt= self.optimizer.count_layers(weights_registered_name)
        self.weights_registered_name = "{}_{}".format(weights_registered_name, cnt)
        self.optimizer.register_params(self.weights_registered_name, self.weights)

    def forward(self, indices):
        self.indices = indices
        self.output = np.take(self.weights, self.indices, axis=0)
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
    def __init__(self, optimizer, normalized_shape, eps=1e-05, data_type=np.float32):
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
        self.gamma = np.ones((1, 1,self.normalized_shape), dtype=self.data_type)
        self.beta = np.zeros((1, 1, self.normalized_shape), dtype=self.data_type)

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
        self.grad_gamma = np.sum(grad * self.x_hat, axis=(0, 1), keepdims=True)
        self.grad_beta = np.sum(grad, axis=(0, 1), keepdims=True)

        # Calculate gradient of x_hat
        dx_hat = grad * self.gamma

        # Calculate gradients of mean and var
        D = self.x.shape[-1]
        dvar = np.sum(dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True)

        # Calculate gradient of x
        dx_hat = dx_hat / np.sqrt(self.var + self.eps)
        dx = dx_hat + (dvar * 2 * (self.x - self.mean) / D) + (dmean / D)

        return dx
    def release_memory(self):
        del self.grad_gamma, self.grad_beta

    def update_weights(self):
        self.gamma = self.optimizer.update(self.gamma, self.grad_gamma, "{}.gamma".format(self.register_name))
        self.beta = self.optimizer.update(self.beta, self.grad_beta, "{}.beta".format(self.register_name))
        
        
        
        
class Linear():
    def __init__(self, in_features, out_features, optimizer, use_bias=True, data_type=np.float32):
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
        sqrt_k = 1. / np.sqrt(self.in_features)
        self.weights = np.random.uniform(-sqrt_k, sqrt_k, (self.in_features, self.out_features)).astype(self.data_type)
        if self.use_bias:
            self.bias = np.random.uniform(-sqrt_k, sqrt_k, self.out_features).astype(self.data_type)

    def zero_grad(self):
        self.grad_weights = np.zeros_like(self.weights)
        if self.use_bias:
            self.grad_bias = np.zeros_like(self.bias)
    
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
        self.grad_weights += np.sum(np.matmul(self.x.transpose(0, 2, 1), grad), axis=0)
        if self.use_bias:
            self.grad_bias += np.sum(grad, axis=tuple(range(grad.ndim - 1)))
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
        return np.maximum(0., x)

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
        e_x = np.exp(x - np.max(x, axis = self.axis, keepdims=True))
        self.y =  e_x / np.sum(e_x, axis = self.axis, keepdims=True)
        del e_x
        return np.nan_to_num(self.y, nan=0.)

    def backward(self, grad_y):
        # https://sgugger.github.io/a-simple-neural-net-in-numpy.html
        grad_x = self.y * (grad_y - (grad_y * self.y).sum(axis=self.axis, keepdims=True))
        del self.y
        return np.nan_to_num(grad_x, nan=0.)
    
    
    
    
    
class Tokenizer:
    def __init__(self, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True):
        self.word_index = {}  # A dictionary to map words to token IDs
        self.index_word = {}  # A dictionary to map token IDs to words
        self.oov_token = oov_token  # Token to use for out-of-vocabulary words
        self.filters = filters  # Characters to filter out from the text
        self.lower = lower  # Convert text to lowercase

    def fit_on_texts(self, texts):
        # Iterate through the input texts and build the vocabulary
        tokens = []  # A list to store tokens from all texts
        for text in texts:
            if self.lower:
                text = text.lower()
            for char in self.filters:
                text = text.replace(char, ' ')
            words = text.split()
            tokens.extend(words)
        
        # Sort tokens by frequency
        word_counts = {}  # A dictionary to store word frequencies
        for word in tokens:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
        
        # Add tokens to the word_index and index_word dictionaries
        for idx, word in enumerate(sorted_words):
            self.word_index[word] = idx + 1  # Reserve 0 for the OOV token
            self.index_word[idx + 1] = word

    def texts_to_sequences(self, texts):
        # Convert input texts to sequences of token IDs
        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            for char in self.filters:
                text = text.replace(char, ' ')
            words = text.split()
            sequence = []
            for word in words:
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                else:
                    sequence.append(self.word_index[self.oov_token])
            sequences.append(sequence)
        return sequences
    def get_vocab(self):
        return len(self.word_index)

    def get_config(self):
        # Return the configuration for the tokenizer (useful for saving the tokenizer)
        return {
            'oov_token': self.oov_token,
            'filters': self.filters,
            'lower': self.lower
        }
        
        

class PadSequences:
    def __init__(self, max_sequence_length, padding_value=0):
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value

    def pad_sequences(self, sequences):
        padded_sequences = []
        for sequence in sequences:
            if len(sequence) >= self.max_sequence_length:
                # If the sequence is longer than the specified length, truncate it
                padded_sequence = sequence[:self.max_sequence_length]
            else:
                # If the sequence is shorter, pad it with the padding value
                padded_sequence = sequence + [self.padding_value] * (self.max_sequence_length - len(sequence))
            padded_sequences.append(padded_sequence)
        return padded_sequences
    
    
    
    
    
class DataLoader:
    def __init__(self, max_sequence_length, padding_value):
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        self.tokenizer = Tokenizer(oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
        self.pad_sequences = PadSequences(max_sequence_length=10, padding_value=0)

    def tokenize_text(self, text):
        # Tokenize the text using the Tokenizer class's fit_to_text and text_to_sequences methods
        Tokenizer.fit_on_texts(text)
        sequences = Tokenizer.texts_to_sequences(text)

        input_sequences = []
        output_sequences = []

        for sequence in sequences:
            for i in range(1, len(sequence)):
                input_seq = sequence[:i]
                output_word = sequence[i]

                input_sequences.append(input_seq)
                output_sequences.append(output_word)


        input_sequences = self.pad_sequences.pad_sequences(input_sequences)
       

        return input_sequences, output_sequences
    