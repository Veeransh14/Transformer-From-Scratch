import cupy as cp
import numpy as np
from encoder import Encoder
from decoder import Decoder
from Dataloader import Dataloader
from loss import CrossEntropy

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
