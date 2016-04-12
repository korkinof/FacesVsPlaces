--
--  Copyright (c) 2016, Dimitrios Korkinof
--  All rights reserved.
--

-- Declare timers and error accumulators
local trnTimer  = torch.Timer()
local loadTimer = torch.Timer()

local loss_epoch, acc_epoch, epochSize, batchNumber

local LR, WD
if opt.regime=='continuous' then
  LR = reverse(torch.logspace(-4,-2,opt.nEpochs))
  WD = torch.ge(LR,5e-3):float():mul(5e-4)
elseif opt.regime=='fixed' then
  local rep = math.ceil(opt.nEpochs/7)
  LR = torch.Tensor{1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5}:repeatTensor(rep,1):transpose(1,2):contiguous():view(7*rep)
  WD = torch.Tensor(7*rep):fill(5e-4)
end

local optimState = {learningRate = LR[1], weightDecay = WD[1], momentum = opt.momentum, learningRateDecay = 0}

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
  print('==> doing epoch on training data:')
  print("==> online epoch # "..epoch)
  
  batchNumber = 0
  
  -- Renew optimstates when starting new regime
  if (epoch%5==0) then
    optimState = {learningRate = LR[epoch], weightDecay = WD[epoch], momentum = opt.momentum, learningRateDecay = 0}
  else
    optimState.learningRate = LR[epoch]
    optimState.weightDecay  = WD[epoch]
  end
  
  cutorch.synchronize()

  -- set the dropouts to training mode
  model:training()

  local tm = torch.Timer()
  loss_epoch = 0
  acc_epoch = 0
  
  if opt.epochSize=='auto' then
    epochSize = math.floor(mainLoader.nTrn/opt.batchSize)
  else
    epochSize = opt.epochSize
  end
  
  for i=1,epochSize do
    local st = (i-1)*opt.batchSize+1
    local nd = math.min(i*opt.batchSize,mainLoader.nTrn)
    local ind = torch.range(st,nd)
    -- queue jobs to data-workers
    if (opt.nThreads~=0) then
      pool:addjob(
        -- the job callback (runs in data-worker thread)
        function()
          trnCount = trnCount + 1
          if trnCount%10==0 then collectgarbage() end
          local img, class = workLoader:loadTrnBatch(ind)
          return img, class
        end,
        -- the end callback (runs in the main thread)
        trainBatch
      )
      if (i%10==0) then
        pool:synchronize()
        collectgarbage()
      end
    else
      -- Unthreaded, usefull for debugging
      local img, class = mainLoader:loadTrnBatch(ind)
      local loadTime = loadTimer:time().real
      trainBatch(img,class)
    end
  end

  pool:synchronize()
  
  cutorch.synchronize()
  
  local f = assert(io.open(opt.saveDir..'train.log','a'))
  local msg = string.format('Epoch:%d|RL:%f|Loss:%.4f|Acc:%.4f|\n',
        epoch,optimState.learningRate,loss_epoch/epochSize,acc_epoch/epochSize)
  f:write(msg)
  f:close()
  
  -- Reshuffle training set for the next epoch
  local permutation = torch.randperm(mainLoader.nTrn)
  permutation = torch.LongTensor(permutation:size(1)):copy(permutation)
  mainLoader:reshuffle(permutation)
  
  torch.save(opt.saveDir..'/dataset.t7',mainLoader)
  
  -- Apply the same permutations in the worker threads
  if (opt.nThreads~=0) then
    pool:specific(true)
    for i=1,opt.nThreads do
      pool:addjob(i,
        -- the job callback (runs in data-worker thread)
        function()
          perm = permutation
          workLoader:reshuffle(perm)
          print(string.format('Worker %d has successfully permuted training set.',tid))
        end
      )
    end
    pool:specific(false)
  end
  
  -- save model
  collectgarbage()
  
  torch.save(opt.saveDir..'model_'..tostring(epoch)..'.t7', model)
  torch.save(opt.saveDir..'optimState_'..tostring(epoch)..'.t7', optimState)
  
end -- of train()
-------------------------------------------------------------------------------------------

-- GPU inputs (preallocate)
local img  = torch.CudaTensor()
local cls = torch.CudaTensor()

local parameters,gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(imgCPU,clsCPU)
  
  local loadTime = loadTimer:time().real
  
  batchNumber = batchNumber + 1
  
  -- Train and time
  trnTimer:reset()
  
  cutorch.synchronize()
  
  img:resize(imgCPU:size()):copy(imgCPU)
  cls:resize(clsCPU:size()):copy(clsCPU)
  
  model:zeroGradParameters()
  -- Forward pass
  local pred = model:forward(img)
  local loss = criterion:forward(pred,cls)
  -- Backward pass
  local dPred = criterion:backward(pred,cls)
  model:backward(img,dPred)
  
  local function feval(x)
    if opt.gradNormalize then
      local b = nBatches
      gradParams:div(b)
      loss = loss/b
    end
    return loss, gradParameters
  end
  optim.sgd(feval,parameters,optimState)
  
  cutorch.synchronize()
  
  local trainTime = trnTimer:time().real
    
  _,pred = pred:max(2)
  pred = pred:squeeze(2)
  local acc = torch.eq(pred,cls):mean()
  
  -- Calculate and accumulate errors
  loss_epoch = loss_epoch + loss
  acc_epoch = acc_epoch + acc
  
  -- Print errors
  msg = string.format('Epoch: [%d][%d/%d]\tTime %.3f Loss %.4f Acc %.4f LR %.0e DataLoadTime %.3f \t',
          epoch, batchNumber, epochSize, trainTime, loss_epoch/batchNumber, acc_epoch/batchNumber,
          optimState.learningRate, loadTime)

  print(msg)
  
  loadTimer:reset()
  
  collectgarbage()
  
end