--
--  Copyright (c) 2015, Dimitrios Korkinof
--  All rights reserved.
--

require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

model   = {}
optimState = {}
trainLogger = {}
valLogger   = {}
epoch = 0

do
  
  local files = paths.dir(opt.saveDir)
  local mind = {}
  local oind = {}
  local tmp Nv=0 Nt=0
  for i,s in ipairs(files) do
    tmp = s:split('_')
    if(tmp[1]=='model') then
      mind[#mind+1] = tonumber(tmp[2]:split('%.')[1]) -- For '.' delimiter you need '%.' because otherwise . represents all characters.
    elseif(tmp[1]=='optimState') then
      oind[#oind+1]   = tonumber(tmp[2]:split('%.')[1])
    end
    if(s=='val.log') then
      local tmp = parseLogFile(opt.saveDir..'val.log')
      if tmp['Epoch'] then
        Nv = torch.Tensor(tmp['Epoch']):max()
      end
    elseif(s=='train.log') then
      local tmp = parseLogFile(opt.saveDir..'train.log')
      if tmp['Epoch'] then
        Nt = torch.Tensor(tmp['Epoch']):max()
      end
    end
  end

  mind = torch.Tensor(mind)
  oind = torch.Tensor(oind)
  if mind:dim()*oind:dim()~=0 then -- then there is something 
    
    local Nm = torch.max(mind)
    local No = torch.max(oind)
    
    -- Some sanity checks
    assert(Nm==No,'The last saved model_xx.t7 does not correspond to the last saved optimState_xx.t7.')
    -- Training and validation runs must differ at most by one
    
    opt.reeval = (Nv+1)==Nt
    
    local fname = opt.saveDir..'model_'..tostring(Nm)..'.t7'
    assert(paths.filep(fname),'Model file not found: '..fname)
    print('=> Loading feature layer from file: models/'..fname..'...')
    model = torch.load(fname)
    
    fname = opt.saveDir..'optimState_'..tostring(No)..'.t7'
    assert(paths.filep(fname),'OptimState file not found: '..fname)
    print('=> Loading optimstate from file: models/'..fname..'...')
    optimState = torch.load(fname)
    
    epoch = Nm
    
  else
    print('Nothing to resume. Starting new run...')
    
    -- for the model creation code, check the models/ folder
    model = createModel()
    
    print('==> Converting feature extractor to CUDA...')
    model = model:cuda()
    
    -- Setup a reused optimization state (for sgd). If needed, reload it from disk
    optimState = {
      learningRate = 0,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      dampening = 0.0,
      weightDecay = opt.weightDecay
    }
    
  end
end

-- Create criterion
print('==> Creating criterion...')
criterion = nn.ClassNLLCriterion()
print('==> Converting criterion to CUDA...')
criterion = criterion:cuda()

print('=> Features')
print(model)

print('=> Criterion')
print(criterion)

collectgarbage()