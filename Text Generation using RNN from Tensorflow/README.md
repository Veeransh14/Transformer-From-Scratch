### RNNs
These are simple models which were previously used as against Transformers and GPT models. These models are not quite efficient and tend to get saturated after certain number of epochs. Training them is not a piece of cake. This is beacuse of Exploding and Vanishing gradients problem and lack of parallizabilty in which the loss produced by the loss function gets saturated to a certain value as a result of which training and updatation of model becomes slow and monotonous 
Here is an example in which we have trained a RNN model but it's accuracy has got saturated.
The loss decreases till the end but after 300 epochs although the loss is decreasing but accuracy remained stuck at 86 percent
This implies that the model was learning but not learning things we want it but it is rather adding noise 


![ScreenshotRNNs](https://github.com/Veeransh14/Transformer-From-Scratch/assets/144168166/bca4a87d-68cb-4432-bf53-70f5c2159a16)


  
Model trained for 4 hours on MacM1 gpu 





