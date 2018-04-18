
This folder contains the files needed for training our residual models. 

The architecture code of each model is in file of the respective model name. For example ResidualModel1.lua contains the architecture code for that model.

-----------------------------------------------------------------

train.lua is the main file for training the networks. It used data.lua for handling and processing the data with multiple threads. It also uses logmessage file in this folder to log the informations and also makes use of custom cost function and several utilities available in other folders.

--------------------------------------

It needs data in hdf5 format. The input images and flow data need to be concatenated along the channel and written as ('/data'+index), representing index of this data. 

------------------------------------
Training the models requires GPU with minimum memory depending on mini batch size and spatial dimension of input. 
To train with the FlyingChairs data of batchsize '8', GPU with 8GB memory would be needed.

-------------------------------------

To run the code:
----------------


th train.lua 'true' 'trainData.h5' 'chair' 200 10 'ResidualModel3'

The parameters are:

1. isTrain : 'true' or 'false'    - 'false' for validation of models

2. dataset : 'trainData.h5' or 'testData.h5' depending on choice of first parameter

3. data mode: 'chair' or 'sintel'

4. number of epochs

5. the snapshot interval to save the model

6. the model (network) to be trained
