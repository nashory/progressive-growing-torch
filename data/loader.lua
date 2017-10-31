-- Multi-threading
-- referenced (https://github.com/soumith/dcgan.torch/blob/master/data)
-- Copyright (c) 2017, Minchul Shin [See LICENSE file for details]

require 'os'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}

local result = {}
local unpack = unpack and unpack or table.unpack


-- option list
-- 'threads', 'batchSize', 'manualSeed'

function data.new(_params)
    local _params = _params or {}
    local self = {}
    for k,v in pairs(data) do
        self[k] = v
    end

    -- get params or set default value.
    local nthreads = _params.nthreads or 1                  -- default thread num is 1.
    local manualSeed = _params.manualSeed or os.time()      -- random seed.
    self.batchSize = _params.batchSize


    --local donkey_file = 'donkey.lua'
    if nthreads > 0 then
        self.threads = Threads( nthreads,
                                function() require 'torch' end,
                                function(thread_id)
                                    opt = _params
                                    local seed = (manualSeed and manualSeed or 0) + thread_id
                                    torch.manualSeed(seed)
                                    torch.setnumthreads(1)
                                    print(string.format('Starting donkey with id: %d, seed: %d',
                                          thread_id, seed))
                                    assert(opt, 'option parameters not given')
                                    require('data.dataloader')
                                    loader=DataLoader(opt)
                                end
                               )
    end

    local nTrainSamples = 0
    local nTestSamples = 0
    self.threads:addjob(    function() return loader:size('train') end,
                            function(c) nTrainSamples = c end)
    self.threads:addjob(    function() return loader:size('test') end,
                            function(c) nTestSamples = c end)
    self.threads:synchronize()
    self._trainlen = nTrainSamples
    self._testlen = nTestSamples

    return self
end

function data._getFromTrainThreads()
    assert(opt.batchSize, 'batchSize not found.')
    return loader:getSample('train')
end

function data._getFromTestThreads()
    assert(opt.batchSize, 'batchSize not found.')
    return loader:getSample('test')
end

function data._pushResult(...)
    local res = {...}
    if res == nil then self.threads:synchronize() end
    table.insert(result, res)
end

function data.table2Tensor(table)
    assert( table[1]:size(1)==3 or table[1]:size(1)==1, 
            'the tensor channel should be 1(gray) or 3(rgb).')
    assert( table[1]~=nil, 'no elements were found in batch table.')
    local len = #table
    local res = torch.Tensor(len, table[1]:size(1), table[1]:size(2), table[1]:size(3)):zero()
    for i = 1, len do res[{{i},{},{},{}}]:copy(table[i]) end
    return res
end

function data.getBatch(self, target)
    assert(target=='train' or target=='test')
    assert(self.batchSize, 'batchSize not found.')
    local dataTable = {}
    ---queue another job
    if target == 'train' then
        for i = 1, self.batchSize do
            self.threads:addjob(self._getFromTrainThreads,
                                function(c) table.insert(dataTable, c) end)
        end
    elseif target == 'test' then
        for i = 1, opt.batchSize do
            self.threads:addjob(self._getFromTestThreads, self._pushResult)
            self.threads:dojob()
        end
    end
    self.threads:synchronize() 
    collectgarbage()
    local res = self.table2Tensor(dataTable)
    return res
end


function data.getSample(self, target)
    assert(target=='train' or target=='test')
    assert(self.batchSize, 'batchSize not found.')
    local dataTable = {}
    if target == 'train' then
        self.threads:addjob(self._getFromTrainThreads, function(c) table.insert(dataTable, c) end)
        self.threads:dojob()
    elseif target == 'test' then
        self.threads:addjob(self._getFromTestThreads, function(c) table.insert(dataTable, c) end)
        self.threads:dojob()
    end
    return unpack(dataTable)
end


function data:size(target)
    if target == 'train' then return self._trainlen
    elseif target == 'test' then return self._testlen
    elseif target == 'all' or target == nil then return (self._trainlen + self._testlen) end
end



return data























