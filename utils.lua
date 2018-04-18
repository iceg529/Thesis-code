--  The first three utility functions are from the NVIDIA DIGITS tool (https://github.com/NVIDIA/DIGITS)
--  Rest of the utitilty functions are our own code 

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'cunn'
require 'cutorch'

package.path = debug.getinfo(1, "S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

-- round function
function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end

-- return whether a Luarocks module is available
function isModuleAvailable(name)
  if package.loaded[name] then
    return true
  else
    for _, searcher in ipairs(package.searchers or package.loaders) do
      local loader = searcher(name)
      if type(loader) == 'function' then
        package.preload[name] = loader
        return true
      end
    end
    return false
  end
end

-- attempt to require a module and drop error if not available
function check_require(name)
    if isModuleAvailable(name) then
        return require(name)
    else
        assert(false,"Did you forget to install " .. name .. " module? c.f. install instructions")
    end
end

-- weights to be used for downsampling the output fields
function getWeight(chns, lnt)
   local weights = torch.Tensor(chns,chns,lnt,lnt)
   local accum_weight = 0
   local ii, jj, tmpWt
   local scale = (lnt-1)/2
   ii = 1
   for i = -scale,scale,1 do
     jj = 1
     for j = -scale,scale,1 do
       tmpWt = (1 -(torch.abs(i)/(scale+1))) * (1 -(torch.abs(j)/(scale+1)))
       weights[{ {1},{1},ii,jj }] = tmpWt
       weights[{ {2},{2},ii,jj }] = tmpWt
       weights[{ {1},{2},ii,jj }] = 0
       weights[{ {2},{1},ii,jj }] = 0
       accum_weight = accum_weight + tmpWt
       jj = jj + 1
     end
     ii = ii + 1
   end
  
   weights:div(accum_weight)
   return weights
end

----- mean normalization -------------
function normalizeMean(meanData, frame1, frame2)
  meanData = meanData:cuda()
  local temp = torch.Tensor(1,meanData[1]:size(1),meanData[1]:size(2),meanData[1]:size(3))
  local temp2 = torch.Tensor(temp:size())
  temp[1]:copy(meanData[1])
  temp2[1]:copy(meanData[3])
  frame1 = torch.add(frame1,-1,temp:expandAs(frame1))
  frame1:cdiv(temp2:expandAs(frame1))
  temp[1]:copy(meanData[2])
  temp2[1]:copy(meanData[4])
  frame2 = torch.add(frame2,-1,temp:expandAs(frame2))
  frame2:cdiv(temp2:expandAs(frame2))
  return frame1, frame2
end

----- fusing mean data -------------
function fuseMean(meanData1,meanData2,width_scaled,height_scaled)
  local N1 = 22232
  local N2 = 904*2  
  local fusedMean = torch.Tensor(4,3,height_scaled,width_scaled)
  fusedMean[1] = ((image.scale(meanData2[1],width_scaled,height_scaled)):mul(N2)):add((image.scale(meanData1[1],width_scaled,height_scaled)):mul(N1)):div(N1 + N2)
  fusedMean[2] = ((image.scale(meanData2[2],width_scaled,height_scaled)):mul(N2)):add((image.scale(meanData1[2],width_scaled,height_scaled)):mul(N1)):div(N1 + N2)
  fusedMean[3] = ((image.scale(meanData2[3],width_scaled,height_scaled)):mul(N2)):add((image.scale(meanData1[3],width_scaled,height_scaled)):mul(N1)):div(N1 + N2)
  fusedMean[4] = ((image.scale(meanData2[4],width_scaled,height_scaled)):mul(N2)):add((image.scale(meanData1[4],width_scaled,height_scaled)):mul(N1)):div(N1 + N2)
  return fusedMean
end

----- color augmentation --------------------
local inTensor = torch.CudaTensor() 
local augParams = torch.CudaTensor()
function colorAugmentation(inTensor,augParams)
  inTensor = inTensor:cuda()

  -- multiplicative color changes per channel
  local tempTensor = torch.CudaTensor()
  tempTensor = inTensor:clone()
  tempTensor[1]:mul(augParams.colFac1)
  tempTensor[2]:mul(augParams.colFac2)
  tempTensor[3]:mul(augParams.colFac3)
  local factor = inTensor:sum()/(tempTensor:sum()+0.001)
  inTensor:mul(factor)

  -- gamma changes
  inTensor:clamp(0,255)
  inTensor:pow(augParams.gamma)  

  -- contrast changes
  local contrastFac = 1-(augParams.contrast)
  local luma = (1 - contrastFac) * (image.rgb2yuv(inTensor))[1]:mean()
  inTensor:mul(contrastFac):add(luma)

  inTensor:clamp(0,255)
  inTensor = inTensor:double()
  return inTensor
end



