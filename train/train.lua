--[[ 
	train.lua : code to train our residual models, can be adapted to train any neural networks
	
	Refer to README file in the folder for details on how to use the code, the acceptable input data structures, the parameters to pass while running the code and limitations.   
	
	The structure of the code for handling data efficiently and training our models, is adapted from the below mentioned source. Mainly Data loader and logger functionalities are used.
    
	*    Title: DIGITS (Deep Learning GPU Training System), NVIDIA Corporation
	*    Author: Yeager et al.,
	*    Date: 2015
       	*    Availability: https://github.com/NVIDIA/DIGITS
--]]

require 'image'
require 'torch'
require 'xlua'
require 'pl'
require 'trepl'
require 'lfs'
require 'nn'
require 'cunn'
require 'cutorch'
require 'optim'
require '../AvgEndPointError' -- custom cost function

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
  package.path = dir_path .."?.lua;".. package.path
end
local opTtrain, opTval
local isTrain = arg[1]

if isTrain == 'true' then
  opTtrain = arg[2]  --'trainData.h5'
else 
  opTval = arg[2]  --'testData.h5'
end 
local opTDataMode = arg[3] -- chair or sintel
local opTthreads = 3 -- number of threads
local opTepoch = tonumber(arg[4]) -- number of epochs
local opTsnapshotInterval = tonumber(arg[5]) -- interval of epochs to save the model
local epIncrement = 0 
local opTsave = "logFiles"  -- location of the saved models and error log
 
profiler = xlua.Profiler(false, true) -- profiler to track the time

require 'logmessage'
require '../utils'
torch.setnumthreads(opTthreads)

----------------------------------------------------------------------

require 'data'
local modelLocation = arg[6]
require (modelLocation) -- 'model'

-- DataLoader objects take care of loading data from
-- databases. Data loading and tensor manipulations
local trainDataLoader, trainSize, inputTensorShape
local valDataLoader, valSize

local num_threads_data_loader = opTthreads
local valData = {}

local meanFile = hdf5.open('../meanDataSintel.h5','r')  -- mean of images from Sintel dataset
local meanData2 = meanFile:read('/data'):all()    
meanFile:close()
meanFile = hdf5.open('../meanData.h5','r')        -- mean of images from FlyingChairs dataset
local meanData1 = meanFile:read('/data'):all()     
meanFile:close()
meanData = meanData1 

if isTrain == 'true' then
	-- create data loader for training dataset
	trainDataLoader = DataLoader:new(
	      num_threads_data_loader, -- num threads
	      package.path,
	      opTtrain,
	      true, -- train
	      false
	)
	-- retrieve info from train DB (number of records and shape of input tensors)
	trainSize, inputTensorShape = trainDataLoader:getInfo()
	logmessage.display(0,'found ' .. trainSize .. ' images in train db' .. opTtrain)
else
	---- create data loader for validation dataset
	valDataLoader = DataLoader:new(
	      1, -- num threads
	      package.path,
	      opTval,
	      false, -- train
	      false
	)
	---- retrieve info from train DB (number of records and shape of input tensors)
	valSize, valInputTensorShape = valDataLoader:getInfo()
	logmessage.display(0,'found ' .. valSize .. ' images in validation db' .. opTval)
	
	if valDataLoader:acceptsjob() then
	  valDataLoader:scheduleNextBatch(valSize, 1, valData, true) 
	end
	valDataLoader:waitNext()
	if opTDataMode == 'sintel' then
	  local widthVal = valData.im1:size(4)
  	  local heightVal = valData.im1:size(3)
  	  local scale_w = math.ceil(widthVal / 32) 
  	  local scale_h = math.ceil(heightVal / 32)
  	  local width_scaled = scale_w * 32
  	  local height_scaled = scale_h * 32
	  meanData = fuseMean(meanData1,meanData2,width_scaled,height_scaled)
	end
        valData.im1, valData.im2 = normalizeMean(meanData, valData.im1, valData.im2)
end

local downSampleFlowWeights = getWeight(2, 7) -- convolution weights , args: 2 - num of channels , 7 - conv filter size
downSampleFlowWeights:cuda()

-- validation function
local function validation(model,valData,criterion,flowWeights)
  --model:evaluate()
  local valErr = 0
  local input, flInput, tmpValImg1, tmpValImg2, tmpValFlow

  -- this scale adaptation is done in many places, to ensure dimensions are a multiple of 32. It is an requirement to pass through the layers of our models.
  local widthVal = valData.im1:size(4)
  local heightVal = valData.im1:size(3)
  local scale_w = math.ceil(widthVal / 32) 
  local scale_h = math.ceil(heightVal / 32)
  local width_scaled = scale_w * 32
  local height_scaled = scale_h * 32
  if ((width_scaled == widthVal) and (height_scaled == heightVal)) then
    input = torch.Tensor(1,2*valData.im1:size(2),valData.im1:size(3),valData.im1:size(4))
    flInput = torch.Tensor(1,2,valData.im1:size(3),valData.im1:size(4))
  else
    input = torch.Tensor(1,2*valData.im1:size(2),height_scaled,width_scaled)
    flInput = torch.Tensor(1,2,height_scaled,width_scaled)
  end
  for i = 1,valData.im1:size(1) do
    if ((width_scaled ~= widthVal) or (height_scaled ~= heightVal)) then
	tmpValImg1 = image.scale(valData.im1[i],width_scaled,height_scaled)
    	tmpValImg2 = image.scale(valData.im2[i],width_scaled,height_scaled) 
    	tmpValFlow = image.scale(valData.flow[i],width_scaled,height_scaled)
    	input[1] = torch.cat(tmpValImg1, tmpValImg2, 1)
	flInput[1] =  tmpValFlow 
    else
	input[1] = torch.cat(valData.im1[i], valData.im2[i], 1) 
	flInput[1] =  valData.flow[i]
    end    
    
    input = input:cuda()
    flInput = flInput:cuda()
    local output = model:forward(input)
    
    -- downsampling of groundtruth to match with output of our models.
    local mod = nn.SpatialConvolution(2,2, 7, 7, 4,4,3,3) 
    mod.weight = flowWeights
    mod.bias = torch.Tensor(2):fill(0)
    mod = mod:cuda()
    local down5 = mod:forward(flInput)
    down5 = down5:cuda()
    
    local module = nn.SpatialUpSamplingBilinear(4):cuda()
    local predFi = module:forward(output)
    local err = criterion:forward(output, down5)
    valErr = valErr + err
    if i == valData.im1:size(1) then    
	print('model validated ' .. i)
    end
  end
  valErr = valErr / valData.im1:size(1)
  collectgarbage()
  return valErr
end
----------------------------------------------------------------------

-- Log results to files
trainLogger = optim.Logger(paths.concat(opTsave, 'train.log'))
trainLogger:setNames{'Training error'}
trainLogger:style{'+-'}
trainLogger:display(false)

valLogger = optim.Logger(paths.concat(opTsave, 'validation.log'))
valLogger:setNames{'Validation error'}
valLogger:style{'+-'}
valLogger:display(false)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode into a 1-dim vector
local model = require('weight-init')(getResModel(), 'kaiming')

model = model:cuda()
local criterion = nn.AvgEndPointError() -- AvgEndPointError
criterion = criterion:cuda()
if model then
   parameters,gradParameters = model:getParameters()
end

---------------------- function to save the intermediate models -----------
local function saveModel(modelToSave, directory, prefix, epoch)
    local filename
    local modelObjectToSave
    if modelToSave.clearState then
        -- save the full model
        filename = paths.concat(directory, prefix .. 'LC1_LR3_' .. epoch + epIncrement .. '_Model.t7')  
        modelObjectToSave = modelToSave:clearState()
    else
        -- this version of Torch doesn't support clearing the model state => save only the weights
        local Weights,Gradients = modelToSave:getParameters()
        filename = paths.concat(directory, prefix .. '_' .. epoch .. '_Weights.t7')
        modelObjectToSave = Weights
    end
    logmessage.display(0,'Snapshotting to ' .. filename)
    torch.save(filename, modelObjectToSave)
    logmessage.display(0,'Snapshot saved - ' .. filename)
end

----------------------

if isTrain == 'true' then
	
        local saveFlag = false
	local currentTrainErr = 10^(7)

	-- train batch size depend on the GPU memory availaility. Choose smaller/higher bacth size according to dimension of data and memory of GPU. 
	local trainBatchSize = 8 
	local logging_check = trainSize 
	local next_snapshot_save = 0.3
	local snapshot_prefix = 'flownet'

	----------------------
	-- epoch value will be calculated for every batch size. To maintain unique epoch value between batches, it needs to be rounded to the required number of significant digits.
	local epoch_round = 0 -- holds the required number of significant digits for round function.
	local tmp_batchsize = trainBatchSize
	while tmp_batchsize <= trainSize do
	    tmp_batchsize = tmp_batchsize * 10
	    epoch_round = epoch_round + 1
	end
	logmessage.display(0,'While logging, epoch value will be rounded to ' .. epoch_round .. ' significant digits')

	------------------------------
	local loggerCnt = 0
	local epoch = 1
        local actualEp = epoch + epIncrement 

	logmessage.display(0,'started training the model')
	local config = {learningRate = (0.0001), 
		           weightDecay = 0.0004, 
		           momentum = 0.9,
		           learningRateDecay = 0 }

	while epoch<=opTepoch do
	  local time = sys.clock()  
	  ------------------------------
	  local NumBatches = 0
	  local curr_images_cnt = 0
	  local loss_sum = 0
	  local loss_batches_cnt = 0
	  local learningrate = 0
	  local im1, im2, flow
	  local dataLoaderIdx = 1
	  local data = {}
	  local input, flowInput
	  print('==> doing epoch on training data:')
	  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. trainBatchSize .. ']')
	  	  
	  local t = 1
	  while t <= trainSize do --trainSize
	    -- disp progress
	    xlua.progress(t, trainSize)
	    local time2 = sys.clock()
	    profiler:start('pre-fetch')
	    -- prefetch data thread
	    --------------------------------------------------------------------------------
	    while trainDataLoader:acceptsjob() do      
	      local dataBatchSize = math.min(trainSize-dataLoaderIdx+1,trainBatchSize)
	      if dataBatchSize > 0 and dataLoaderIdx < math.floor(trainSize/trainBatchSize)+1 then   
		trainDataLoader:scheduleNextBatch(dataBatchSize, dataLoaderIdx, data, true)
		dataLoaderIdx = dataLoaderIdx + 1 --dataBatchSize
	      else break end
	    end
	    NumBatches = NumBatches + 1

	    -- wait for next data loader job to complete
	    trainDataLoader:waitNext()
	    --------------------------------------------------------------------------------
	    profiler:lap('pre-fetch')
	    -- get data from last load job
	    local thisBatchSize = data.batchSize
	    im1 = torch.Tensor(data.im1:size())
	    im2 = torch.Tensor(data.im2:size())
	    flow = torch.Tensor(data.flow:size())
	    im1:copy(data.im1)
	    im2:copy(data.im2)
	    flow:copy(data.flow)

 	    local widthVal = im1:size(4)
  	    local heightVal = im1:size(3)
   	    local scale_w = math.ceil(widthVal / 32) 
  	    local scale_h = math.ceil(heightVal / 32)
  	    local width_scaled = scale_w * 32
  	    local height_scaled = scale_h * 32

	    ----- mean normalization -------------
	    if opTDataMode == 'sintel' then
	      meanData = fuseMean(meanData1,meanData2,width_scaled,height_scaled)
	    end
	    im1, im2 = normalizeMean(meanData, im1, im2)	    
	    
	    if ((width_scaled ~= widthVal) or (height_scaled ~= heightVal)) then
		local tmpValImg1 = torch.Tensor(im1:size(1),im1:size(2),height_scaled, width_scaled)
	        local tmpValImg2 = torch.Tensor(im2:size(1),im2:size(2),height_scaled, width_scaled)
	        local tmpValFlow = torch.Tensor(flow:size(1),flow:size(2),height_scaled, width_scaled)
		for i = 1,im1:size(1) do 
	          tmpValImg1[i] = image.scale(im1[i],width_scaled,height_scaled)
    	          tmpValImg2[i] = image.scale(im2[i],width_scaled,height_scaled) 
    	          tmpValFlow[i] = image.scale(flow[i],width_scaled,height_scaled)
	        end
    	        input = torch.cat(tmpValImg1, tmpValImg2, 2)
	        flowInput =  tmpValFlow
	    else
		input = torch.cat(im1, im2, 2)
	        flowInput = flow
	    end
	    
	    profiler:start('training process')
	    ------------------------------------------------------------------------------------------------------------------------------
	    -- create closure to evaluate f(X) and df/dX
	    local feval = function(x)
	      -- get new parameters
	      if x ~= parameters then
		parameters:copy(x)
	      end

	      -- reset gradients
	      gradParameters:zero()

	      -- f is the average of all criterions
	      local f = 0
              profiler:start('feval process')
	      local output
	      -- remove from this line till above for loop
	      flowInput = flowInput:cuda()
              input = input:cuda()		
	      
 	      output = model:forward(input)
		
	      local mod = nn.SpatialConvolution(2,2, 7, 7, 4,4,3,3) -- nn.SpatialConvolution(2,2,1, 1, 4, 4, 0, 0)
	      mod.weight = downSampleFlowWeights
	      mod.bias = torch.Tensor(2):fill(0)
	      mod = mod:cuda()
	      local down5 = mod:forward(flowInput)
	      down5 = down5:cuda()
	      local err = criterion:forward(output, down5) --grdTruth
	      f = f + err
		
	      print(f)
	      profiler:lap('feval process')
	      -- return f and df/dX
	      return f,gradParameters
	    end
	    
	    -- optimize on current mini-batch	    		         
	    _, train_err = optim.adam(feval, parameters, config)
	    -----------------------------------------------------------------------------------------------------------------------------
	    profiler:lap('training process')

	    -------------------------------------BLOCK TO CHECK LATER---------------------------------------------------------------------    
	    profiler:start('logging process')
	    -- adding the loss values of each mini batch and also maintaining the counter for number of batches, so that average loss value can be found at the time of logging details
	    loss_sum = loss_sum + train_err[1]
	    loss_batches_cnt = loss_batches_cnt + 1
	    
	    local current_epoch = (epoch-1)+round((math.min(t+trainBatchSize-1,trainSize))/trainSize, epoch_round)
	    
	    print(current_epoch)
	    
	    print(loggerCnt)
	    -- log details on first iteration, or when required number of images are processed
	    curr_images_cnt = curr_images_cnt + thisBatchSize
	    
	    -- update logger/plot
	    if (epoch==1 and t==1) or curr_images_cnt >= logging_check then      
	      local avgLoss = loss_sum / loss_batches_cnt 
	      
	      trainLogger:add{avgLoss}
	      trainLogger:plot()
	      if ((currentTrainErr - avgLoss) < (10^(-5))) then
	        saveFlag = true
		break
	      end
	      currentTrainErr = avgLoss

	      --logmessage.display(0, 'Training (epoch ' .. current_epoch)
	      if (epoch==1 and t==1) then 
		curr_images_cnt = thisBatchSize
	      else
		curr_images_cnt = 0 -- For accurate values we may assign curr_images_cnt % logging_check to curr_images_cnt, instead of 0
 		loss_sum = 0
	        loss_batches_cnt = 0
	      end	      
	      loggerCnt = loggerCnt + 1
	    end

	    if current_epoch >= next_snapshot_save then
	      --model:double() 
	      saveModel(model, opTsave, snapshot_prefix, current_epoch)
	      --model:cuda() 
	      next_snapshot_save = (round(current_epoch/opTsnapshotInterval) + 1) * opTsnapshotInterval -- To find next epoch value that exactly divisible by opt.snapshotInterval
	      last_snapshot_save_epoch = current_epoch
	    end
	    -------------------------------------------------------------------------------------------------------------------------------
	    
	    t = t + thisBatchSize
	    profiler:lap('logging process')
	    print('The data loaded till index ' .. data.indx)
	    if math.fmod(NumBatches,10)==0 then
	      collectgarbage()
	    end
	  end
	  if saveFlag then
	    saveModel(model, opTsave, snapshot_prefix, current_epoch)
	    break
	  end
	  ------------------------------
	  -- time taken
	  time = sys.clock() - time
	  --time = time / trainSize
	  print("==> time to learn for 1 epoch = " .. (time) .. 's')
	   
	  epoch = epoch+1
	  actualEp = actualEp+1
	end     	
else
  
  -- Cross validate the models with test data. Validation is not a part of training process because of memory reasons. 
  -- Automated cross validation to make choices on hyperparameters is better and can try with smaller batch size or use a larger GPU
 
  local models = {'../models/ResidualModel1','../models/ResidualModel4'} -- many models together can be validated on a testset
 
  for i=1,#models do --for i=1,6 do
    local model1 = torch.load(models[i] .. '.t7') -- or use models saved from intermediate logfiles.
    local avgValErr = validation(model1, valData, criterion, downSampleFlowWeights)
    valLogger:add{avgValErr}
    valLogger:plot()
  end
end

-- enforce clean exit
os.exit(0)
