
This folder contains the files needed for evaluating our residual models on individual examples. 

The trained models need to be downloaded first. Use the script 'download_models.py' in previous folder to download the models. 
ResidualModel1.t7 contains the trained ResidualModel1 with learned weights.

-----------------------------------------------------------------

test.lua is the main file for training the networks. It also used other utilities available in the folders.

--------------------------------------

It saves the output flow data as 'flowOut.flow' . We can also give the desired flow file name as an optional parameter in the end. 

------------------------------------
Evaluating the models requires GPU with minimum memory depending on the model and spatial dimension of input. 
To test with the FlyingChairs and Sintel data, GPU with 6GB memory would be more than enough.

-------------------------------------

To run the code:
----------------


th test.lua ../models/ResidualModel4.t7 img1.png img2.png

The parameters are:

1. the model file : location of the trained residual model

2. Image 1 

3. Image 2 


