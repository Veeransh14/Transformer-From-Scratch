# Transformer-From-Scratch


## Overview

This repository contains an implementation of the Transformer architecture from scratch, written in Python .py files. 

The Transformer is a powerful neural network architecture that has been shown to achieve state-of-the-art performance on a wide range of natural language processing tasks, including language modeling, machine translation, and sentiment analysis.

# About the Project

## Aim
The goal of this project is to gain a deeper understanding of how the Transformer works and how it can be applied to different natural language processing tasks.

## Description
NLP can be performed using many Architectures and model but we have started our journey with Neural Networks as they form the basis of each and every NLP model. Here we have implemented RNNs and LSTMs models and considering it's drawbacks we have shifted towards more advanced models like Transformer.
By implementing the Transformer from scratch, we can get a hands-on understanding of the key components of the architecture, including multi-head self-attention, feedforward layers, and layer normalization. We would be feeding various datasets to our model such that it generates some text based on that dataset on which it has been trained.

## Tech Stack
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [Python](https://www.python.org/)
- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/doc/#)
- [Google Collab](https://colab.research.google.com/)

# File Structure



# Projects we worked on
- Image Classification
- Spam Detection using Logistic Regression 
- MNIST digit classifier
- RNNs
- Transformer
- Building Transformer Package Using PyPi

## Image Classification
- This project deals with Binary Classification of Objects and would produce result 1(True) or 0(False).
- In our project we had to classify between cat and non-cat images.
- We used tensorflow and keras predefined libraries to achieve our goal.

## Spam Detection using Logistic Regression 
- The idea of this post is to understand step by step working of the spam filter model. Also, next time when you see a “You have won a lottery” SMS rather than 
  ignoring it, you might prefer to report it as a spam.
- Here the logistic regression model is implemented from scratch, including functions for initialization, propagation, optimization, and prediction.

## MNIST digit Classifier
- MNIST digit classification is a fundamental problem in machine learning and computer vision. The trained neural network can accurately identify digits (0 to 9) 
  from images of handwritten characters.
- The MNIST dataset is loaded using Keras, ReLU activation functions are used for hidden layers, and softmax activation is used for the output layer.

## RNNs
These are simple models which were previously used as against Transformers and GPT models. These models are not quite efficient and tend to get saturated after certain number of epochs. Training them is not a peice of cake. This is beacuse of Exploding and Vanishing gradients problem in which the loss produced by the loss function gets saturated to a certain value as a result of which training and updatation of model becomes slow and monotonous.
Here is an example in which we have trained a RNN model but it's loss has got saturated.
![ScreenshotRNNs](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/bca4a87d-68cb-4432-bf53-70f5c2159a16)

This is our RNN model- in which we can clearly see how it's giving arbitrary and trash results.

https://github.com/Veeransh14/Transformer-From-Scratch/blob/main/.rnn%20implementation.ipynb

# Transformer Architecture
The Transformer architecture consists of a set of encoders and a decoders, In the paper they used 6 of each.

The encoder processes the input sequence of tokens, while the decoder generates the output sequence.

Here's a brief overview of each component:

### Input Embedding

The input sequence of tokens is first embedded into a continuous vector space using an embedding layer. 

This layer maps each token to a fixed-length vector that captures its semantic meaning. The embedding layer is trained jointly with the rest of the model.

### Positional Encoding

Since the Transformer does not use recurrent or LSTM layers, it needs a way to incorporate positional information about the tokens. This is done using a positional encoding layer, which adds a set of sinusoidal (sin and cos) functions to the input embeddings. The frequencies and phases of these functions encode the position of each token in the input sequence.

### Encoder Layers

The encoder consists of multiple identical layers, each of which applies a set of operations to the input sequence in parallel. The core operation of each encoder layer is multi-head self-attention (they used 8 heads in the paper), which allows the model to attend to different parts of the input sequence at different levels of granularity. 

The outputs of the self-attention layer are passed through a feedforward neural network, and the resulting representations are combined with the original input sequence using residual connections and layer normalization.

### Decoder Layers

The decoder also consists of multiple identical layers, each of which applies a similar set of operations to the output sequence. In addition to the self-attention and feedforward layers, each decoder layer also includes a multi-head attention layer that attends to the encoder output. This allows the decoder to align the input and output sequences and generate more accurate translations.

### Output Layer
The final layer of the decoder is a softmax layer that produces a probability distribution over the output vocabulary. During training, the model is trained to maximize the likelihood of the correct output sequence. During inference, the output sequence is generated by sampling from the predicted probability distribution.

This is our Transformer Architecture with Encoder on left and Decoder on right.
![Screenshottrans](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/3e8016db-97ff-4e99-9349-2a5350484833)






## Building Transformer Package Using PyPi
- Added one special feature in which we can directly import our Transformer Architecture just like we import predefined libraries in python. This has been achieved using PyPi 
  which is Python Package Index.


# Future Works
- We are planning to work on real time transformer models like Llama and many others too.
- Generative Pre-Trained Transformer models that is GPT is our next topic of discussion and research.
- Planning to modify, optimize and fine tune our model to give more accurate predictions. We would be achieveing this by comparing our model with real time world Transformer 
  Architecture Models.



# Contributers
- [Kshitij Shah](https://github.com/kshitijdshah99)
- [Mayank Palan](https://github.com/MayankPalan2004)
- [Veeransh Shah](https://github.com/Veeransh14)

  

# Acknowledgement and Resources
- Coursera Andrew NG courses-1,2,5 https://www.coursera.org/specializations/deep-learning
- Andrej Karpathy https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ 
- Stanford CS224N https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- Research Paper-Attention Is All You Need https://arxiv.org/abs/1706.03762
- Special thanks to our Mentor [Labeeb Asari](https://github.com/labeeb-7z) who mentored us selflessly these 2 long months



