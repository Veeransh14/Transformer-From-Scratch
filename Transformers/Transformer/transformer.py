#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Encoder import Encoder
from Decoder import Decoder

class Transformer():
    def __init__(self, optimizer, source_vocab_size, target_vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type):
        self.encoder = Encoder(optimizer, source_vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type)
        self.decoder = Decoder(optimizer, target_vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type)
        self.data_type = data_type

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


# In[ ]:





# In[ ]:




