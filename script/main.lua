require 'nn'

--require 'cunn'
local opts = require 'script.opts'
local network = require 'models.network'

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


-- generator / discriminator parameter setting.
g_config = {
            ['num_channels']=3,
            ['fmap_base']=4096,
            ['fmap_decay']=1.0,
            ['fmap_max']=512,
            ['transition_tick']=opt.transition_tick,
            ['nz']=512,
            ['normalize_latents']=true,
            ['use_weightnorm']=false,
            ['use_pixelwise']=true,
            ['use_leakyrelu']=true,
            ['use_batchnorm']=true,
            ['use_tanh']=true,
            }
d_config = {
            ['num_channels']=3,
            ['resolution']=32,
            ['fmap_base']=4096,
            ['fmap_decay']=1.0,
            ['fmap_max']=512,
            ['use_wscale']=true,
            ['fmap_gdrop']=true,
            ['normalize_latents']=true,
            ['use_weightnorm']=false,
            ['use_pixelwise']=true,
            ['use_leakyrelu']=true,
            ['use_batchnorm']=true,
}

config = {
            ['D']=d_config,
            ['G']=g_config,
}

-- load players(gen, dis)
local gan_models = {}
local gan_gen = network.get_init_gen(g_config)
local gan_dis = network.get_init_dis(d_config)
print ('Generator structure: ')    
print(gan_gen)
print ('Discriminator structure: ')    
print(gan_dis)



gan_models = {gan_gen, gan_dis}


-- loss metrics
local gan_criterion = {nn.MSECriterion()}


-- run trainer
require 'script.pggan'
local optimstate = {}
local gan_trainer = PGGAN(gan_models, gan_criterion, opt, optimstate, config)
gan_trainer:train(myloader)


print('Congrats! You just finished the training.')













  
