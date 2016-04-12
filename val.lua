--
--  Copyright (c) 2016, Dimitrios Korkinof
--  All rights reserved.
--

-- Declare timers and error accumulators
local valTimer  = torch.Timer()
local loadTimer = torch.Timer()

local loss_total, acc_total, batchNumber

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function val()
  print('==> doing epoch on training data:')
  print("==> online epoch # "..epoch)
  
  batchNumber = 0
  
  cutorch.synchronize()

  -- set the dropouts to training mode
  model:evaluate()

  local tm = torch.Timer()
  loss_total = 0
  acc_total = 0
  
  for i=1,math.ceil(mainLoader.nVal/opt.valBatchSize) do
    local st = (i-1)*opt.valBatchSize+1
    local nd = math.min(i*opt.valBatchSize,mainLoader.nVal)
    local ind = torch.range(st,nd)
    -- queue jobs to data-workers
    if (opt.nThreads~=0) then
      pool:addjob(
        -- the job callback (runs in data-worker thread)
        function()
          valCount = valCount + 1
          if valCount%10==0 then collectgarbage() end
          local img, class = workLoader:loadValBatch(ind)
          return img, class
        end,
        -- the end callback (runs in the main thread)
        evalBatch
      )
      if (i%10==0) then
        pool:synchronize()
        collectgarbage()
      end
    else
      -- Unthreaded, usefull for debugging
      local img, class = mainLoader:loadValBatch(ind)
      local loadTime = loadTimer:time().real
      evalBatch(img,class)
    end
  end

  pool:synchronize()
  
  cutorch.synchronize()
  
  local f = assert(io.open(opt.saveDir..'val.log','a'))
  local msg = string.format('Epoch:%d|Loss:%.4f|Acc:%.4f|\n',epoch,loss_total/batchNumber,acc_total/batchNumber)
  f:write(msg)
  f:close()
  
  -- save model
  collectgarbage()
  
end -- of train()
-------------------------------------------------------------------------------------------

-- GPU inputs (preallocate)
local img = torch.CudaTensor()
local cls = torch.CudaTensor()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function evalBatch(imgCPU,clsCPU)
  
  local loadTime = loadTimer:time().real
  
  batchNumber = batchNumber + 1
  
  -- Train and time
  valTimer:reset()
  
  cutorch.synchronize()
  
  img:resize(imgCPU:size()):copy(imgCPU)
  cls:resize(clsCPU:size()):copy(clsCPU)
  cls = cls:repeatTensor(10,1):transpose(2,1):contiguous():view(10*clsCPU:size(1))
  
  model:zeroGradParameters()
  -- Forward pass
  local pred = model:forward(img)
  local loss = criterion:forward(pred,cls)
  
  cutorch.synchronize()
  
  local valTime = valTimer:time().real
  
  pred = pred:view(clsCPU:size(1),10,opt.nCat)
  pred = pred:sum(2):squeeze()
  
  _,pred = pred:max(2)
  pred = pred:squeeze(2):float()
  local acc = torch.eq(pred,clsCPU):float():mean()
  
  -- Calculate and accumulate errors
  loss_total = loss_total + loss
  acc_total = acc_total + acc
  
  -- Print errors
  msg = string.format('Validation: [%d][%d/%d]\tTime %.3f Loss %.4f Acc %.4f DataLoadTime %.3f \t',
          epoch, batchNumber, math.ceil(mainLoader.nVal/opt.valBatchSize), valTime,
          loss_total/batchNumber, acc_total/batchNumber,loadTime)

  print(msg)
  
  loadTimer:reset()
  
  collectgarbage()
  
end