--
--  Copyright (c) 2015, Dimitrios Korkinof
--  All rights reserved.
--

-- The following is uncommented when debugging

require 'torch'
require 'cutorch'
require 'paths'
require 'optim'
require 'nn'
require 'lfs'
require 'options/opts'
require 'dataset'
require 'image'
require 'util'

opt = options()

if opt.debug then require('mobdebug').start() end

torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(opt.GPU) -- by default, use GPU 1

torch.manualSeed(opt.seed) -- Fix seeds for repeatability

print('Saving everything to: ' .. opt.saveDir)
if not paths.filep(opt.saveDir..'/dataset.t7') then
  os.execute('mkdir -p ' .. opt.saveDir)
end

-- Including correct version of model script
local config = opt.netType ..'_'.. opt.backend
print('=> Including model script from file: models/'..config..'.lua') 
paths.dofile('models/'..config..'.lua')

mainLoader = dataset()
torch.save(opt.saveDir..'/dataset.t7',mainLoader)

-- Initialise thread pool of workers for data loading
paths.dofile('pool_init.lua')

paths.dofile('autoresume.lua')

paths.dofile('train.lua')
if opt.test then paths.dofile('test.lua') end
paths.dofile('val.lua')

if opt.reeval then
  val()
end

epoch = epoch + 1

for i=1,opt.nEpochs do
  
  train()  
  val()
  -- Option to perform testing or not
  if opt.test then test() end
  
  epoch = epoch + 1
end
