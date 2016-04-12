--
--  Copyright (c) 2015, Dimitrios Korkinof
--  All rights reserved.
--
local Threads = require 'threads'

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at dataset.lua
-------------------------------------------------------------------------------
do -- start K datathreads (workers)
  if opt.nThreads > 0 then
    local Options = opt -- make an upvalue to serialize over to donkey threads
    pool = Threads(
      opt.nThreads,
      function()
        require 'torch'
        require 'options/opts'
        require 'dataset'
        require 'image'
        gm = assert(require 'graphicsmagick')
      end,
      function(idx)
        opt = options()
        tid = idx
        trnCount=0
        valCount=0
        tstCount=0
        print(string.format('Starting worker thread with id: %d.', tid))
        -- Load dataset file or initialise for the worker thread
        datasetObjPath = opt.saveDir..'/dataset.t7'
        assert(paths.filep(datasetObjPath),string.format('Thread %d failed to load the dataset file: %s.',tid,datasetObjPath))
        workLoader = torch.load(datasetObjPath)
        print(string.format('File loaded correctly, nTrn=%d',workLoader.nTrn))
      end
    );
  else -- single threaded data loading. useful for debugging
    pool = {}
    workLoader = dataLoader
    function pool:addjob(f1, f2) f2(f1()) end
    function pool:synchronize() end
  end
end
pool:synchronize()
