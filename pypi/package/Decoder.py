
import numpy as np

from BaseLayer import Embedding
from BaseLayer import Dropout
from BaseLayer import Linear
from BaseLayer import Softmax
from CombinedLayer import PositionalEncoding
from Encoder import DecoderBlock
from BaseLayer import LayerNormalization
from CombinedLayer import MultiHeadAttention
from CombinedLayer import PositionWiseFeedForward

class Decoder():
    def __init__(self, optimizer, vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type):
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model, optimizer, data_type)
        self.dropout = Dropout(dropout_rate, data_type)
        self.positional_encoding = PositionalEncoding(max_len, d_model, data_type)
        self.decoder_layers = [DecoderBlock(optimizer, d_model, d_ff, num_attention_heads, dropout_rate, data_type) for _ in range(block_num)]
        self.fc = Linear(d_model, vocab_size, optimizer, True, data_type)
        self.softmax = Softmax()

    def forward(self, target, source, target_mask, src_tgt_mask, training):
        target = self.embedding.forward(target) * np.sqrt(self.d_model)
        target = self.positional_encoding.forward(target)
        target = self.dropout.forward(target, training)
        for decoder_layer in self.decoder_layers:
            target =  decoder_layer.forward(target, source, target_mask, src_tgt_mask, training)
        target = self.fc.forward(target)
        target = self.softmax.forward(target)
        return target

    def backward(self, grad):
        
     
        grad = self.softmax.backward(grad)
        
        grad = self.fc.backward(grad)
        
        self.grad_source_sum = 0
        for decoder_layer in reversed(self.decoder_layers):
            grad, grad_source = decoder_layer.backward(grad)
            self.grad_source_sum += grad_source
        grad = self.dropout.backward(grad)
        grad = self.positional_encoding.backward(grad) * np.sqrt(self.d_model)
        grad = self.embedding.backward(grad)
        return grad

    def update_weights(self):
        self.fc.update_weights()
        for decoder_layer in reversed(self.decoder_layers):
            decoder_layer.update_weights()
        self.embedding.update_weights()
        
        
        

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
        