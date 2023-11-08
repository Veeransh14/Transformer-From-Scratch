
import cupy as cp
import numpy as np 

from Tokenizer import Tokenizer  
from PadSequences import PadSequences  
from encoder import Encoder
from decoder import Decoder
from Dataloader import Dataloader
from loss import CrossEntropy



class DataLoader:
    def __init__(self, max_sequence_length, padding_value):
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        self.tokenizer = Tokenizer(oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
        self.pad_sequences = PadSequences(max_sequence_length=10, padding_value=0)

    def tokenize_text(self, text):
        # Tokenize the text using the Tokenizer class's fit_to_text and text_to_sequences methods
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)

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


class CrossEntropy():
    def __init__(self, padding_id, vocab_size):
        self.eps = 1e-6
        self.padding_id = padding_id
        self.vocab_size = vocab_size

    def one_hot(self, label):
        label_num = len(label)
        one_hot = cp.zeros((label_num, self.vocab_size))
        one_hot[cp.arange(label_num), label] = 1
        self.one_hot_label = one_hot
        return one_hot

    def forward(self, pred, label):
        one_hot_label = self.one_hot(label)
        loss = -(one_hot_label * cp.log(pred + self.eps)).sum(-1)
        return loss

    def grad(self, pred, label):
        grad = pred - self.one_hot_label
        return grad

class MSE():
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self):
        loss = ((self.y_pred - self.y_true) ** 2).mean()
        return loss

    def grad(self):
        grad = 2 * (self.y_pred - self.y_true)
        return grad
    
    
    
class Adam():
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.98, eps=1e-9, warmup_steps=4000, d_model=512):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.registered_layer_params = {}

    def _step(self):
        self.step += 1

    def set_lr(self):
        self.lr = self.d_model ** -0.5 * min((self.step+1) ** -0.5, (self.step+1) * self.warmup_steps ** -1.5)

    def save_params(self, params_name, t, mean, var):
        self.registered_layer_params[params_name] = {
                "t": t,
                "mean": mean,
                "var": var
            }

    def register_params(self, params_name, params):
        if params_name not in self.registered_layer_params:
            self.save_params(params_name=params_name, t=0, mean=cp.zeros_like(params), var=cp.zeros_like(params))
        else:
             print("NOOOOOOOOO!")

    def count_layers(self, layer_name):
        cnt = 0
        for key in self.registered_layer_params.keys():
            if key.startswith(layer_name):
                cnt += 1
        return cnt

    def update(self, param, param_grad, params_name):
        
        # print('Updating ', params_name)
        # print('param: ', param.mean())
        # print(params_name, 'param_grad: ', cp.abs(param_grad).sum())

        assert param.shape == param_grad.shape, 'param shape: {}, param_grad shape: {}'.format(param.shape, param_grad.shape)
        t = self.registered_layer_params[params_name]["t"]
        mean = self.registered_layer_params[params_name]["mean"]
        var = self.registered_layer_params[params_name]["var"]
        # update
        t += 1
        mean = self.beta1 * mean + (1 - self.beta1) * param_grad
        var = self.beta2 * var + (1 - self.beta2) * param_grad ** 2
        self.save_params(params_name, t, mean, var)
        mean_hat = mean / (1 - self.beta1 ** t)
        var_hat = var / (1 - self.beta2 ** t)
        self.set_lr()
        delta = self.lr * mean_hat / (cp.sqrt(var_hat) + self.eps)

        param = param - delta
        return param

class SGD():
    def __init__(self, lr=1e-3):
        self.lr = lr

    def update(self, param, param_grad):
        return param - self.lr * param_grad
    
    
    

class Transformer():
    def __init__(self, optimizer, source_vocab_size, target_vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type,padding_value):
        self.encoder = Encoder(optimizer, source_vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type)
        self.decoder = Decoder(optimizer, target_vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type)
        self.data_type = data_type
        self.data_loader = DataLoader(max_sequence_length=max_len, padding_value=padding_value)
        self.cross_entropy = CrossEntropy(padding_value, vocab_size=target_vocab_size)
        self.padding_id=padding_value
        self.optimizer=optimizer

    def forward(self, source_ids, target_ids, source_mask, target_mask, src_tgt_mask,training=True):
        encoder_output = self.encoder.forward(source_ids, source_mask, training)
        decoder_output = self.decoder.forward(target_ids, encoder_output, target_mask, src_tgt_mask, training)
        return decoder_output
    def backward(self, grad):
        grad = self.decoder.backward(grad)
        grad = self.encoder.backward(self.decoder.grad_source_sum)
        return grad

    def update_weights(self):
        self.decoder.update_weights()
        self.encoder.update_weights()
        
        
    def get_padding_mask(self,ids, padding_id):
         
        batch_size, seq_len = ids.shape
        mask1d = (ids != padding_id).astype(int)
        mask_cnt = mask1d.sum(-1)
        mask = cp.zeros((batch_size, seq_len, seq_len), cp.int8)
        for i in range(batch_size):
            mask[i, :mask_cnt[i], :mask_cnt[i]] = 1
            
        return mask
  
    

    def get_subsequent_mask(self,ids):
   
        seq_len = ids.shape[1]
        mask = cp.tril(cp.ones((seq_len, seq_len)), k=0).astype(int)
        return mask

    def get_src_tgt_mask(self,src_ids, tgt_ids, padding_id):
        batch_size, src_seq_len = src_ids.shape
        _, tgt_seq_len = tgt_ids.shape
        src_mask_cnt = (src_ids != padding_id).astype(int).sum(-1)
        tgt_mask_cnt = (tgt_ids != padding_id).astype(int).sum(-1)
        mask = cp.zeros((batch_size, tgt_seq_len, src_seq_len), cp.int8)
        for i in range(batch_size):
            mask[i, :tgt_mask_cnt[i], :src_mask_cnt[i]] = 1
        return mask
        
    def fit(self,num_epochs, batch_size,input_sequences,output_sequences,loss):
        #input and output sequence from dataloader
        a=len(input_sequences)//batch_size
        a=a*batch_size
        input_sequences=input_sequences[:a]
        output_sequences=output_sequences[:a]
        input_sequences = cp.array(input_sequences)
        output_sequences = cp.array(output_sequences)

        for epoch in range(1, num_epochs):
           
            shuffled_indices = cp.random.permutation(len(input_sequences))
            epoch_loss = 0  

            for i in range(0, len(input_sequences), batch_size):
                batch_indices = shuffled_indices[i:i + batch_size]
                batch_input = input_sequences[batch_indices]
                batch_target = output_sequences[batch_indices]
                source = batch_input
                target_in = batch_target

                source_mask = self.get_padding_mask(source, padding_id)
                target_mask =self.get_padding_mask(target_in, padding_id) & get_subsequent_mask(target_in)
                src_tgt_mask = self.get_src_tgt_mask(source, target_in, padding_id)

        # Forward pass
                predictions = self.forward(source, target_in, source_mask, target_mask, src_tgt_mask, training=True)
                output = predictions
                output_shape = output.shape
        

                output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])
                loss = self.cross_entropy.forward(output, target_in.flatten()).mean().get()

        # Accumulate the loss for this batch
                epoch_loss += loss

                grad = self.cross_entropy.grad(output,target_in.flatten()).reshape(output_shape)
        

        # Backpropagation
                grad = self.backward(grad)


        # Update model weights
                self.update_weights()
                self.optimizer._step()
                
               

            # Calculate the average loss for the epoch
            average_epoch_loss = epoch_loss / (len(input_sequences) / batch_size)


            print("Epoch {} - Average Loss: {}".format(epoch, average_epoch_loss))
            
            
            


    