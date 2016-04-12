--
--  Copyright (c) 2016, Dimitrios Korkinof
--  All rights reserved.
--

require 'torch'
require 'cutorch'
require 'paths'
require 'nn'
require 'inn'
require 'cudnn'
require 'image'
require 'options/opts'
local gm = assert(require 'graphicsmagick')
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)

opt = options()

local data = torch.load(paths.concat(opt.dataDir,'data.t7'))

local function loadImage(path)
  
  local oH  = opt.sampleSize
  local oW  = opt.sampleSize
  local out = torch.Tensor(10,3,oW,oH)

  local img = gm.Image():load(path,opt.loadSize,opt.loadSize)
  
  -- find the smaller dimension, and resize it to 250 (while keeping aspect ratio)
  local iW, iH = img:size()
  if iW < iH then
    img:size(opt.loadSize,opt.loadSize*iH/iW);
  else
    img:size(opt.loadSize*iW/iH,opt.loadSize);
  end
  iW, iH = img:size();
  img = img:toTensor('float','RGB','DHW')
  
  -- Augment image into 10 crops
  local w1 = math.ceil((iW-oW)/2)
  local h1 = math.ceil((iH-oH)/2)
  out[1] = image.crop(img, w1, h1, w1+oW, h1+oW) -- center patch
  out[2] = image.hflip(out[1])
  h1 = 1; w1 = 1;
  out[3] = image.crop(img, w1, h1, w1+oW, h1+oW)  -- top-left
  out[4] = image.hflip(out[3])
  h1 = 1; w1 = iW-oW;
  out[5] = image.crop(img, w1, h1, w1+oW, h1+oW)  -- top-right
  out[6] = image.hflip(out[5])
  h1 = iH-oH; w1 = 1;
  out[7] = image.crop(img, w1, h1, w1+oW, h1+oW)  -- bottom-left
  out[8] = image.hflip(out[7])
  h1 = iH-oH; w1 = iW-oW;
  out[9] = image.crop(img, w1, h1, w1+oW, h1+oW)  -- bottom-right
  out[10] = image.hflip(out[9])
  
  -- Normalise across loaded batch
  for i=1,3 do -- channels
    out[{{},{i},{},{}}]:add(-data.mean[i])
    out[{{},{i},{},{}}]:div(data.std[i])
  end

  return out
end

local test_path  = 'test/'
local model_path = paths.concat(opt.saveDir,'model_1.t7')

assert(paths.filep(model_path),string.format('Model file not found: %s',model_path))
print('=> Loading feature layer from file: models/'..model_path..'...')

model = torch.load(model_path)
model:evaluate()

local img = torch.CudaTensor()
for f in paths.files(test_path,'.jpg') do
  
  local img = loadImage(paths.concat(test_path,f))
    
  img = img:cuda()
  
  local pred = model:forward(img)
  pred = pred:sum(1):squeeze()
  
  local _,pos = pred:max(1)
  pos = pos:squeeze()
  
  print(string.format('Image %s was classified as %s.',f,data.category[pos]))
  
  local imgCPU = image.drawText(img[1]:float(), data.category[pos], 10, 20,{color = {0, 0, 255}, bg = {0, 255, 0}, size = 4})
  image.display(imgCPU)

  
end