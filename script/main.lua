require 'nn'
require 'cunn'
local opts = require 'script.opts'
--local gen = require 'models.gen'
--local dis = require 'models.dis'

local gen = require 'models.original.gen'
local dis = require 'models.original.dis'

-- basic settings.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(2)


-- get opt ans set seed.
local opt = opts.parse(arg)
print(opt)
if opt.seed == 0 then   opt.seed = torch.random(1,9999) end
torch.manualSeed(opt.seed)
print(string.format('Seed : %d', opt.seed))

-- set gpu.
if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.manualSeedAll(opt.seed)
    cutorch.setDevice(opt.gpuid+1)          -- lua index starts from 1 ...
end

-- create dataloader.
local loader = paths.dofile('../data/data.lua')


-- import trainer script.
require 'script.began'

-- load players(gen, dis)
local began_models = {}
local began_dis = dis.create_model(opt.sampleSize, opt)         -- discriminator
local began_gen = gen.create_model(opt.sampleSize, opt)         -- generator    
began_models = {began_gen, began_dis}
print ('BEGAN generator : ')    
print(began_gen)
print ('BEGAN discriminator(enc/dec) : ')
print(began_dis)

--loss metrics
local began_criterion = {nn.AbsCriterion()}
--local began_criterion = {nn.SmoothL1Criterion()}

-- run trainer
local optimstate = {}
local began_trainer = BEGAN(began_models, began_criterion, opt, optimstate)
began_trainer:train(10000, loader)


print('Congrats! You just finished the training.')













  
