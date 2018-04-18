require 'torch'
require 'hdf5'
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'xlua'
loadfile('../AvgEndPointError.lua')()
loadfile('../AvgAngularError.lua')()
loadfile('../utils.lua')()

profiler = xlua.Profiler(false, true)

local num_parameters, model, img1, img2, flow, width, height, scale_w, scale_h, width_scaled, height_scaled
local is_scaled = false

for k, v in ipairs(arg) do
  num_parameters = k
end

if num_parameters < 3 then
  print('Error : Missing mandatory arguments')
else
  local sample = image.load(arg[2],3,'byte')
  img1 = torch.Tensor(1,sample:size(1),sample:size(2),sample:size(3))
  img2 = torch.Tensor(1,sample:size(1),sample:size(2),sample:size(3))  
  img1[1]:copy(sample):cuda()
  img2[1]:copy(image.load(arg[3],3,'byte')):cuda()

  width = sample:size(3)
  height = sample:size(2)
  local shape = width * height
  local max_shape = 802432
  if shape > max_shape then
    width = math.ceil((max_shape / shape) * width)
    height = math.ceil((max_shape / shape) * height)
    is_scaled = true
  end

  scale_w = math.ceil(width / 32) 
  scale_h = math.ceil(height / 32)
  width_scaled = scale_w * 32
  height_scaled = scale_h * 32
  
  if is_scaled then
    print("To fit with the model memory constraints, input dimensions are changed to " .. height_scaled .. "*" .. width_scaled)
  end

  local meanFile = hdf5.open('../meanDataSintel.h5','r')  --meanData.h5
  local meanData2 = meanFile:read('/data'):all()    
  meanFile:close()
  meanFile = hdf5.open('../meanData.h5','r')
  local meanData1 = meanFile:read('/data'):all()     
  meanFile:close()
  local meanData = fuseMean(meanData1,meanData2,width_scaled,height_scaled)

  if arg[4] then
    flow = arg[4]
  else
    flow = 'flowOut.flo'
  end   
  
  local img1_scaled = torch.Tensor(1,3,height_scaled,width_scaled)
  local img2_scaled = torch.Tensor(1,3,height_scaled,width_scaled)
  img1_scaled[1] = image.scale(img1[1],width_scaled,height_scaled)
  img2_scaled[1] = image.scale(img2[1],width_scaled,height_scaled)

  img1_scaled, img2_scaled = normalizeMean(meanData, img1_scaled, img2_scaled)

  local input = torch.cat(img1_scaled, img2_scaled, 2):cuda() 
  model = torch.load(arg[1])  

  local pred = model:forward(input)

  local module = nn.SpatialUpSamplingBilinear(4):cuda()
  local predFinal = module:forward(pred:cuda())
  local predFinal_scaled = torch.CudaTensor(1,2,height,width)
  if (is_scaled == false) and ((width_scaled ~= width) or (height_scaled ~= height)) then
    predFinal_scaled[1] = image.scale((predFinal[1]):double(),width,height)
    torch.save('flowOutput.t7',predFinal_scaled[1])
    width_scaled = width
    height_scaled = height
  else
    torch.save('flowOutput.t7',predFinal[1])
  end
  os.execute("python writeFloFile.py " .. flow .. " flowOutput.t7 " .. width_scaled .. " " .. height_scaled) 
end


