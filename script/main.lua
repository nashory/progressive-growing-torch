require 'nn'
require 'cunn'
local opts = require 'script.opts'
local gen = require 'models.gen'
local dis = require 'models.dis'

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
require 'script.pggan'

-- generator / discriminator parameter setting.
g_config = {
            ['num_channels']=3,
            ['resolution']=32,
            ['label_size']=0,
            ['fmap_base']=4096,
            ['fmap_decay']=1.0,
            ['fmap_max']=512,
            ['nz']=512,
            ['normalize_latents']=true,
            ['use_wscale']=true,
            ['use_pixelnorm']=true,
            ['use_leakyrelu']=true,
            ['use_batchnorm']=false,
            ['use_tanh']=true,
            }
d_config = {
            ['num_chanels']=1,
            ['resolution']=32,
            ['label_size']=0,
            ['fmap_base']=4096,
            ['fmap_decay']=1.0,
            ['fmap_max']=256,
            ['use_wscale']=true,
            ['fmap_gdrop']=true,
            ['fmap_layernorm']=false,
}



-- load players(gen, dis)
local gan_models = {}
--local gan_dis = dis.create_model(opt.sampleSize, d_config)         -- discriminator
local gan_gen = gen.create_model(g_config)         -- generator    
--gan_models = {gan_gen, gan_dis}
print ('Generator structure: ')    
print(gan_gen)
--print ('Discriminator structure: ')
--print(gan_dis)

--loss metrics
--local gan_criterion = {nn.MSECriterion()}

-- run trainer
--local optimstate = {}
--local gan_trainer = P_G_GAN(gan_models, gan_criterion, opt, optimstate)
--gan_trainer:train(10000, loader)


print('Congrats! You just finished the training.')













  
