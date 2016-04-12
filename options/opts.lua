--
--  Copyright (c) 2015, Dimitrios Korkinof
--  All rights reserved.
--

local obj = torch.class('options')

function obj:__init()
  
  self.defaultDir = '/media/korkinof/Data/Research/Datasets'                    -- Default dataset directory
  self.netType    = 'vgg_f'                                                    -- Model to be used
  self.dataset    = 'FvP'                                                       -- Name of the dataset
  self.dataDir    = self.defaultDir ..'/'..self.dataset                         -- Home of eBay dataset
  self.annotFile  = self.defaultDir ..'/'..self.dataset..'/annotation.txt'      -- Annotation file
  
  ------------ General options --------------------
  self.seed        = 2                 -- Manually set RNG seed
  self.GPU         = 1                 -- Default preferred GPU')
  self.backend     = 'cudnn'           -- Options: cudnn | fbcunn | cunn
  self.test        = false             -- Option not to perform testing
  self.debug       = false             -- Debugging
  ------------- Data options ------------------------
  self.nThreads   = 5                  -- number of data loading threads
  self.nCat       = 2                  -- Number of classes, excluding the background class
  self.loadSize   = 250                
  self.sampleSize = 224                
  ------------- Training options --------------------
  self.nEpochs        = 15                    -- Number of total epochs to run
  self.epochSize      = 'auto'                -- Number of batches per epoch | auto
  self.batchSize      = 32                     -- mini-batch size (1 = pure stochastic)
  self.gradNormalize  = false                 -- Normalise gradients with batch size
  self.loadPretrained = 'data/vgg_f_imagenet.mat'    -- path | vgg_D_imagenet.mat , vgg_D_23.t7
  self.regime         = 'fixed'
  ------------- Training options --------------------
  self.valBatchSize   = 5           -- Validation batch size
  ---------- Optimization options ----------------------
  self.momentum    = 0.9               -- momentum
  self.weightDecay = 5e-4              -- weight decay
  ---------- Model options ----------------------------------
  self.autoresume  = true              -- provide path to model to retrain with
  
  self.saveDir = self.defaultDir ..'/runs_FvP/'..self.netType..'/' -- Results directory
  
  if self.debug then
    self.nThreads  = 0
    self.epochSize = 5
  end
  
end

return obj