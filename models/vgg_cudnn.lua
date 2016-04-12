require 'cudnn'

local modelType = 'A'

local cfg = {}

-- Create tables describing VGG configurations A, B, D, E
if modelType == 'A' then
  cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512}
elseif modelType == 'B' then
  cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512}
elseif modelType == 'D' then
  cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512}
  elseif modelType == 'E' then
  cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512}
else
  error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
end
  
local sz_conv_standard = 0
local step_standard = 1    -- all image reductions
local offset0 = 0          -- summing-up all convolutional kernel widths
local offset  = 0          -- Adding zero-padding times the dimension doubling

 -- forward
for i=1,#cfg do
  if cfg[i] == 'M' then
    step_standard = 2*step_standard
  else
    offset0 = offset0 + 3
  end
end
-- reverse
for i=#cfg,1,-1 do
  if cfg[i] == 'M' then
    sz_conv_standard = 2*sz_conv_standard
  else
    sz_conv_standard = sz_conv_standard + 1
  end
end
offset = sz_conv_standard/2

function createModel()
  
  local model = nn.Sequential()
  local iChannels = 3;
  do
    for k,v in ipairs(cfg) do
      if v == 'M' then
        model:add(cudnn.SpatialMaxPooling(2,2,2,2))
      else
        local oChannels = v;
        local conv3 = cudnn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
        model:add(conv3)
        model:add(cudnn.ReLU(true))
        iChannels = oChannels;
      end
    end
  end
  
  local ft_H,ft_W = featureMapSize(opt.sampleSize,opt.sampleSize)
  
  model:add(nn.View(512*ft_H*ft_W))
  model:add(nn.Linear(512*ft_H*ft_W, 4096))
  model:add(cudnn.ReLU(true))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(4096, 4096))
  model:add(cudnn.ReLU(true))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(4096,opt.nCat))
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

function featureMapSize(iH,iW)
  oH = iH
  oW = iW
  for i=1,#cfg do
    if cfg[i] == 'M' then
      local padW, padH, dW, dH, kW, kH
      padW=0; padH=0; dW=2; dH=2; kW=2; kH=2
      oH = torch.floor((oH+2*padH-kH)/dH+1)
      oW = torch.floor((oW+2*padW-kW)/dW+1)
    else
      local padW, padH, dW, dH, kW, kH
      padW=1; padH=1; dW=1; dH=1; kW=3; kH=3
      oH = torch.floor((oH+2*padH-kH)/dH+1)
      oW = torch.floor((oW+2*padW-kW)/dW+1)
    end
  end
  return oH,oW
end
