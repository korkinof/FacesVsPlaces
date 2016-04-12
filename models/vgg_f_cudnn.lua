require 'cudnn'
require 'inn'
function createModel()

  local model = nn.Sequential() -- branch 1
  model:add(cudnn.SpatialConvolution(3,64,11,11,4,4,0,0))       -- 224 -> 55
  model:add(cudnn.ReLU(true))
  model:add(inn.SpatialSameResponseNormalization(3,0.00005,0.75))
  model:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
  
  model:add(cudnn.SpatialConvolution(64,256,5,5,1,1,2,2))       -- 27 ->  27
  model:add(cudnn.ReLU(true))
  model:add(inn.SpatialSameResponseNormalization(3,0.00005,0.75))
  model:add(cudnn.SpatialMaxPooling(3,3,2,2))                  -- 27 ->  13
  
  model:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      -- 13 ->  13
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      -- 13 ->  13
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      -- 13 ->  13
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

  model:add(nn.View(256*5*5))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(256*5*5, 4096))
  model:add(nn.Threshold(0, 1e-6))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(4096, 4096))
  model:add(nn.Threshold(0, 1e-6))
  model:add(nn.Linear(4096, opt.nCat))
  model:add(nn.LogSoftMax())
  
  model = require('weight-init')(model,'xavier_caffe')
  if opt.loadPretrained=='none' then
    print('No pretrained model to load, initialising randomly.')
  else
    local extension = opt.loadPretrained:split('%.')[2]
    if extension=='t7' then
      model = loadPretrainedTorch(model,opt.loadPretrained)
    elseif extension=='mat' then
      model = loadPretrainedMatlab(model,opt.loadPretrained)
    else
      error('Unknown extension of the pretrained model file.')
    end
  end
  
  collectgarbage()

  return model
end