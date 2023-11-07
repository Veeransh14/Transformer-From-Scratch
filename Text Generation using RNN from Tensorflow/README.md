### RNNs
These are simple models which were previously used as against Transformers and GPT models. These models are not quite efficient and tend to get saturated after certain number of epochs. Training them is not a peice of cake. This is beacuse of Exploding and Vanishing gradients problem in which the loss produced by the loss function gets saturated to a certain value as a result of which training and updatation of model becomes slow and monotonous.
Here is an example in which we have trained a RNN model but it's loss has got saturated.
![ScreenshotRNNs](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/bca4a87d-68cb-4432-bf53-70f5c2159a16)

This is our RNN model- in which we can clearly see how it's giving arbitrary and trash results.


