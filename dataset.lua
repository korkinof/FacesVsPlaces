--
--  Copyright (c) 2015, Dimitrios Korkinof
--  All rights reserved.
--

require 'torch'
require 'math'
require 'xlua'
local gm = assert(require 'graphicsmagick')
torch.setdefaulttensortype('torch.FloatTensor')

local dataset = torch.class('dataset')

function dataset:__init()
  local dataFile = paths.concat(opt.dataDir,'data.t7')
  if paths.filep(dataFile) then
    local data = torch.load(dataFile)
    self.category = data.category
    self.imgName  = data.imgName
    self.imgClass = data.imgClass
    self.trnSet = data.trnSet
    self.valSet = data.valSet
    -- self.tstSet = data.tstSet
    self.mean = data.mean
    self.std  = data.std
    
    opt.nCat = #self.category
    self.nImages = #self.imgName
    -- Split to train/validation/test sets
    self.nTrn = math.ceil(0.85*self.nImages)
    self.nVal = self.nImages-self.nTrn
    self.nTst = 0
  else
    self.category = {}
    self.imgName = {}
    self.imgClass = {}
  
    local k = 1
    for d in paths.iterdirs(opt.dataDir) do
      table.insert(self.category,d)
      local n = 1
      for f in paths.files(paths.concat(opt.dataDir,d),'.jpg') do
        if n<14000 then
          table.insert(self.imgName,f)
          table.insert(self.imgClass,k)
        end
        n=n+1
      end
      k=k+1
    end
    
    opt.nCat = #self.category
    
    self.nImages = #self.imgName
    self.imgClass = torch.Tensor(self.imgClass)
    
    -- Split to train/validation/test sets
    self.nTrn = math.ceil(0.85*self.nImages)
    self.nVal = self.nImages-self.nTrn
    self.nTst = 0
    
    -- Reshufle dataset
    torch.manualSeed(opt.seed)
    local perm = torch.randperm(self.nImages)
    
    -- Split dataset
    self.trnSet = perm[{{1,self.nTrn}}]
    self.valSet = perm[{{self.nTrn+1,self.nTrn+self.nVal}}]
    -- self.tstSet = perm[{{self.nTrn+self.nVal+1,self.nImages}}]
    
    -- Calculate mean and std
    self.nSamples = math.min(10000,self.nTrn)
    self.mean,self.std = self:getImMoments()
    
    local data = {}
    data.category = self.category
    data.imgName  = self.imgName
    data.imgClass = self.imgClass
    data.trnSet = self.trnSet
    data.valSet = self.valSet
    -- data.tstSet = self.tstSet
    data.mean = self.mean
    data.std  = self.std
    torch.save(dataFile,data)
  end
end

function dataset:reshuffle(perm)  
  -- Reshuffle training set
  self.trnSet = self.trnSet:index(1,perm)
end

function dataset:loadTrnBatch(ind)
  
  local N = ind:size(1)
  
  local img  = torch.Tensor(N,3,opt.sampleSize,opt.sampleSize)
  local cls = torch.Tensor(N):fill(0)
  
  for i=1,N do
    local pos = self.trnSet[ind[i]]
    img[i],cls[i] = self:loadTrnImage(pos)
  end
  
  -- Normalise across loaded batch
  for i=1,3 do -- channels
    img[{{},{i},{},{}}]:add(-self.mean[i])
    img[{{},{i},{},{}}]:div(self.std[i])
  end
  
  return img,cls
end

function dataset:loadValBatch(ind)
  local N = ind:size(1)
  
  local img  = torch.Tensor(10*N,3,opt.sampleSize,opt.sampleSize)
  local cls = torch.Tensor(N):fill(0)
  
  for i=1,N do
    pos = self.valSet[ind[i]]
    img[{{10*(i-1)+1,10*i},{},{},{}}],cls[i] = self:loadTstImage(pos)
  end
  
  -- Normalise across loaded batch
  for i=1,3 do -- channels
    img[{{},{i},{},{}}]:add(-self.mean[i])
    img[{{},{i},{},{}}]:div(self.std[i])
  end
  
  return img,cls
end

function dataset:loadValBatch(ind)
  local N = ind:size(1)
  
  local img  = torch.Tensor(10*N,3,opt.sampleSize,opt.sampleSize)
  local cls = torch.Tensor(N):fill(0)
  
  for i=1,N do
    local pos = self.valSet[ind[i]]
    img[{{10*(i-1)+1,10*i},{},{},{}}],cls[i] = self:loadTstImage(pos)
  end
  
  -- Normalise across loaded batch
  for i=1,3 do -- channels
    img[{{},{i},{},{}}]:add(-self.mean[i])
    img[{{},{i},{},{}}]:div(self.std[i])
  end
  
  return img,cls
end

function dataset:getImMoments()
  
  local tm = torch.Timer()
  
  print('Estimating the mean and std per channet over '..self.nSamples..' randomly sampled training images')
  
  local meanEstimate = torch.Tensor(3):fill(0)
  local varEstimate  = torch.Tensor(3):fill(0)
  
  for i=1,self.nSamples do
    local img,_ = self:loadTrnImage(self.trnSet[i])
    for j=1,3 do
      meanEstimate[j] = meanEstimate[j] + torch.mean(img[j])
      varEstimate[j]  = varEstimate[j]  + torch.mean(torch.cmul(img[j],img[j]))
    end
    xlua.progress(i,self.nSamples)
  end
  
  meanEstimate = meanEstimate/self.nSamples
  varEstimate  = varEstimate/self.nSamples
  
  mean = meanEstimate
  std  = torch.sqrt(varEstimate-torch.cmul(meanEstimate,meanEstimate))

  print('Time to estimate:',tm:time().real)
  
  return mean,std
end

function dataset:loadTrnImage(pos)
  
  local cls = self.imgClass[pos]
  local path = paths.concat(opt.dataDir,self.category[cls],self.imgName[pos])
  
  -- load image with size hints
  local input = gm.Image():load(path,opt.loadSize,opt.loadSize)
  -- find the smaller dimension, and resize it to 256 (while keeping aspect ratio)
  local iW, iH = input:size()
  if iW < iH then
    input:size(250,250*iH/iW)
  else
    input:size(250*iW/iH,250)
  end
  iW, iH = input:size();
  
  -- Do random crop
  local oW  = opt.sampleSize
  local oH  = opt.sampleSize
  local h1  = math.ceil((iH-oH)/2)
  local w1  = math.ceil((iW-oW)/2)
  local out = input:crop(oW, oH, w1, h1)

  -- Do hflip with probability 0.5
  if torch.uniform() > 0.5 then 
    out:flop()
  end
  
  out = out:toTensor('float','RGB','DHW')
  
  return out,cls
end

function dataset:loadTstImage(pos)
  
  local cls = self.imgClass[pos]
  local path = paths.concat(opt.dataDir,self.category[cls],self.imgName[pos])
  
  local oH  = opt.sampleSize
  local oW  = opt.sampleSize
  local out = torch.Tensor(10,3,oW,oH)

  local img = gm.Image():load(path,opt.loadSize,opt.loadSize)
  -- find the smaller dimension, and resize it to 256 (while keeping aspect ratio)
  local iW, iH = img:size()
  if iW < iH then
    img:size(250,250*iH/iW);
  else
    img:size(250*iW/iH,250);
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

  return out,cls
end

return dataset
