--
--  Copyright (c) 2015, Dimitrios Korkinof
--  All rights reserved.
--

function isnan(n)
  return tostring(n) == tostring(0/0)
end

function meshgrid(x,y)
  local xx = torch.repeatTensor(x, y:size(1),1)
  local yy = torch.repeatTensor(y:view(-1,1), 1, x:size(1))
  return xx:view(xx:size(1)*xx:size(2)), yy:view(yy:size(1)*yy:size(2))
end

function reduce(src,pos)
  local dest
  if src:dim()==3 then
    dest = torch.Tensor(src:size(1),src:size(3)):typeAs(src)
    for i=1,src:size(1) do
      dest[{i,{}}] = src[{i,pos[i],{}}]
    end
  elseif src:dim()==2 then
    dest = torch.Tensor(src:size(1)):typeAs(src)
    for i=1,src:size(1) do
      dest[i] = src[i][pos[i]]
    end
  end
  return dest
end

function expand(src,pos1,pos2,dims)
  local dest = torch.Tensor(dims):typeAs(src):zero()
  for i=1,pos1:size(1) do
    dest[{pos1[i],pos2[i],{}}] = src[{i,{}}]
  end
  return dest
end

function reverse(list)
  local ind = torch.LongTensor(list:size(1)):copy(torch.range(list:size(1),1,-1))
  return list:index(1,ind)
end

function modulus(a,b)
  local res = torch.add(a,-1,torch.floor(a/b)*b)
  res[torch.eq(res,0)] = b
  return res
end

function ind2sub(idx)
  local res
  if idx:sum()>0 then
    local N = idx:size(1)
    local sb  = torch.range(1,N)[idx]
    res = torch.LongTensor(sb:size(1)):copy(sb)
  else
    res = torch.LongTensor()
  end
  return res
end

function parseLogFile(fname)
  local f = assert(io.open(fname,"r"),'Log file could not be openned.')
  local log = {}
  local i=1
  local line = f:read()
  while type(line)=='string' do
    line = line:split('|')
    for j=1,#line do
      local str = line[j]:split(':')
      assert(#str==2,'Log file not in the correct format.')
      if log[str[1]]==nil then log[str[1]]={} end
      log[str[1]][#log[str[1]]+1] = tonumber(str[2])
    end
    i=i+1
    line = f:read()
  end
  f:close()
  return log
end

function sanitize(net)
  local list = net:listModules()
  for _,val in ipairs(list) do
    for name,field in pairs(val) do
      if torch.type(field) == 'cdata' then val[name] = nil end
      if name == 'homeGradBuffers' then val[name] = nil end
      if name == 'input_gpu' then val['input_gpu'] = {} end
      if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
      if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
      if (name == 'output' or name == 'gradInput') then
        val[name] = field.new()
      end
    end
  end
end

function loadPretrainedTorch(model,filePath)
  if paths.filep(filePath) then
    pretrained = torch.load(filePath)
    for i=1,model:size() do
      srcType  = moduleType(pretrained:get(i))
      destType = moduleType(model:get(i))
      if srcType=='ignore' and destType=='ignore' then
        -- Do nothing
      elseif srcType==destType then
        model:get(i).weight:copy(pretrained:get(i).weight:float():contiguous())
        model:get(i).bias:copy(pretrained:get(i).bias:float():contiguous())
      else
        error('Pretrained model to be loaded does not agree with destination.')
      end
    end
    print(string.format('Successfully initialised using pretrained model: %s.',filePath))
  else
    print(string.format('Pretrained model %s was not found.',filePath))
  end
  return model
end


-- The assumption is that pretrained model is smaller or equal to the destination model
function loadPretrainedMatlab(model,filePath)  
  if(paths.filep(filePath)) then
    -- Load Matlab weights file
    local matio = require 'matio'
    matio.use_lua_strings = true -- Read strings as Lua string, instead of character tensors
    local pretrained = matio.load(filePath)
    
    print(string.format('Loading pretrained model: %s....',path))
    local i = 1
    local j = 1
    while i<=model:size() and j<=#pretrained.layers do
      local dstType = moduleType(model:get(i))
      local srcType = moduleType(pretrained.layers[j])
      if srcType=='ignore' and dstType=='ignore' then
        i = i + 1
        j = j + 1
      elseif srcType==dstType then
        model:get(i).weight:copy(pretrained.layers[j].weights[1]:transpose(1,4):transpose(2,3):contiguous())
        model:get(i).bias:copy(pretrained.layers[j].weights[2]:contiguous())
        i = i + 1
        j = j + 1
      elseif srcType=='ignore' then
        j = j + 1
      elseif dstType=='ignore' then
        i = i + 1
       else
        i = i + 1
        j = j + 1
      end
    end
    print(string.format('Successfully initialised using pretrained model: %s.',path))
  else
    print(string.format('Pretrained model %s was not found.',path))
  end
  collectgarbage()
  return model
end

function loadZeilerFromMatlab(model,filePath)
  assert(paths.filep(filePath),'Pretrained model file not found: '..filePath)
  local matio = require 'matio'
  local mat = matio.load(filePath)
  local idx = 1
  for i=1,model:size() do
    if torch.typename(model:get(i))=='nn.SpatialConvolutionMM' or
      torch.typename(model:get(i))=='nn.SpatialConvolution' or
      torch.typename(model:get(i))=='cudnn.SpatialConvolution' then
      model:get(i).weight:copy(mat['conv'..idx..'_w']:transpose(1,4):transpose(2,3))
      model:get(i).bias:copy(mat['conv'..idx..'_b'])
      idx = idx + 1
    elseif torch.typename(model:get(i))=='nn.Linear' then
      model:get(i).weight:copy(mat['fc'..idx..'_w']:transpose(1,2))
      model:get(i).bias:copy(mat['fc'..idx..'_b'])
      idx = idx + 1
    end
  end
end

function moduleType(mod)
  local modType
  if type(mod['type'])=='function' then -- Torch layer
    if    torch.typename(mod)=='nn.SpatialConvolutionMM' 
       or torch.typename(mod)=='nn.SpatialConvolution'
       or torch.typename(mod)=='cudnn.SpatialConvolution' then
      modType = 'conv'..tostring(mod.weight:size(3))
    elseif torch.typename(mod)=='nn.Linear' then
      modType = 'linear'..tostring(mod.weight:size(1))..'x'..tostring(mod.weight:size(2))
    else
      modType = 'ignore'
    end
  elseif type(mod['type'])=='string' then -- Matlab weights
    if string.sub(mod.name,1,4) == 'conv' then
      modType = 'conv'..tostring(mod.weights[1]:size(1))
    elseif string.sub(mod.name,1,2) == 'fc' then
      modType = 'linear'..tostring(mod.weights[1]:size(3))..'x'..tostring(mod.weights[1]:size(4))
    else
      modType = 'ignore'
    end
  end
  return modType
end

function upvalues()
  local variables = {}
  local idx = 1
  local func = debug.getinfo(2, "f").func
  while true do
    local ln, lv = debug.getupvalue(func, idx)
    if ln ~= nil then
      variables[ln] = lv
    else
      break
    end
    idx = 1 + idx
  end
  return variables
end

function locals()
  local variables = {}
  local idx = 1
  while true do
    local ln, lv = debug.getlocal(2, idx)
    if ln ~= nil then
      variables[ln] = lv
    else
      break
    end
    idx = 1 + idx
  end
  return variables
end

