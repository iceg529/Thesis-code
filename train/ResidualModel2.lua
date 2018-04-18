require 'torch'
require 'nn'
require 'nngraph'

function getResModel()
  
  local outputs = {}
  --table.insert(inputs, nn.Identity()())
  local imgIn = nn.Identity()()
--  local flowIn = nn.Identity()()
  local inputs = {imgIn}

  -- stage 1 : filter bank -> squashing -> filter bank -> squashing
  local h1 = imgIn - nn.SpatialConvolution(6, 64, 7, 7, 2, 2, 3, 3)
                   - nn.ReLU()
                   - nn.SpatialConvolution(64, 64, 5, 5, 2, 2, 2, 2)
                   - nn.ReLU()

  -- stage 2 : filter bank -> squashing -> filter bank -> squashing
  local h2 = h1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                - nn.ReLU()
                - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
		- nn.ReLU()

  local h2_1 = nn.CAddTable()({h2, h1})
               

  -- stage 3 : filter bank -> squashing -> filter bank -> squashing
  local h3 = h2_1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h3_1 = nn.CAddTable()({h3, h2_1})               
       
  -- stage 4 : filter bank -> squashing -> filter bank -> squashing
  local h4_0 = h3_1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  
  local h4 = h4_0 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h4_1 = nn.CAddTable()({h4, h3_1})               
       
  -- stage 5 : filter bank -> squashing -> filter bank -> squashing
  local h5 = h4_1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h5_1 = nn.CAddTable()({h5, h4_1})             

  -- stage 6 : filter bank -> squashing -> filter bank -> squashing
  local h6 = h5_1 - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)
  		  - nn.ReLU()

  local h6_0 = nn.CAddTable()({h6, h5_1})               

  local h6_1 = h6_0 - nn.SpatialMaxPooling(2, 2, 2, 2)

  local h6_2 = h6_1 - nn.Copy(cudaTensor, cudaTensor)
		    - nn.MulConstant(0)
  local h6_concat = nn.JoinTable(2)({h6_1, h6_2})

  -- additions Res 2
  local flowDisp1    = h6_0 - nn.SpatialConvolution(64, 2, 3, 3, 1, 1, 1, 1)

  --------------------------
  
  -- stage 7 : filter bank -> squashing -> filter bank -> squashing
  local h7 = h6_1 - nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h7_1 = nn.CAddTable()({h7, h6_concat})               
       
  -- stage 8 : filter bank -> squashing -> filter bank -> squashing
  local h8 = h7_1 - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()
                  - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h8_1 = nn.CAddTable()({h8, h7_1})               
       
  -- stage 9 : filter bank -> squashing -> filter bank -> squashing
  local h9_0 = h8_1 - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
                  - nn.ReLU()

  local h9 = h9_0 - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
		  - nn.ReLU()

  local h9_1 = nn.CAddTable()({h9, h8_1})                

  -- stage 10 : filter bank -> squashing -> filter bank -> squashing
  local h10 = h9_1 - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
                   - nn.ReLU()
                   - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
		   - nn.ReLU()

  local h10_1 = nn.CAddTable()({h10, h9_1})      

  -- stage 11 : filter bank -> squashing -> filter bank -> squashing
  local h11 = h10_1 - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()
                    - nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h11_1 = nn.CAddTable()({h11, h10_1})                
	--	- nn.SpatialMaxPooling(2, 2, 2, 2)
  local h11_2 = h11_1 - nn.Copy(cudaTensor, cudaTensor)
		      - nn.MulConstant(0)
  local h11_concat = nn.JoinTable(2)({h11_1, h11_2})

  -- additions Res 2
  local flowDisp2    = h11_1 - nn.SpatialConvolution(128, 2, 3, 3, 1, 1, 1, 1)
                             - nn.SpatialUpSamplingBilinear(2)

  ---------------------------

  -- stage 12 : filter bank -> squashing -> filter bank -> squashing
  local h12 = h11_1 - nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()
                    - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h12_1 = nn.CAddTable()({h12, h11_concat})                

  -- stage 13 : filter bank -> squashing -> filter bank -> squashing
  local h13 = h12_1 - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()
                    - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h13_1 = nn.CAddTable()({h13, h12_1})                
       
  -- stage 14 : filter bank -> squashing -> filter bank -> squashing
  local h14_0 = h13_1 - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()

  local h14 = h14_0 - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h14_1 = nn.CAddTable()({h14, h13_1})                
       
  -- stage 15 : filter bank -> squashing -> filter bank -> squashing
  local h15 = h14_1 - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()
                    - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h15_1 = nn.CAddTable()({h15, h14_1})
  
  -- commented till h16_1 for Res 5              
  -- stage 16 : filter bank -> squashing -> filter bank -> squashing
  local h16 = h15_1 - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
                    - nn.ReLU()
                    - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
		    - nn.ReLU()

  local h16_1 = nn.CAddTable()({h16, h15_1})

  ---------------------------
        
  -- additions Res 2
  local flowDisp3    = h16_1 - nn.SpatialConvolution(256, 2, 3, 3, 1, 1, 1, 1)
                             - nn.SpatialUpSamplingBilinear(2)

  local Con5 = nn.CAddTable()({flowDisp1, flowDisp2, flowDisp3})
  

  table.insert(outputs, Con5)

  return nn.gModule(inputs, outputs)
end


