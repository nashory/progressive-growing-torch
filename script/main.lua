require 'nn'

--require 'cunn'
local opts = require 'script.opts'
--local gen = require 'models.gen'
local network = require 'models.network'
--local dis = require 'models.dis'

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



--[[
-- create dataloader.
local myloader = require 'script.myloader'
myloader.l_config.batchSize = 54
myloader:renew(myloader.l_config)
local batch = myloader:getBatch('train')
print(batch:size())


myloader.l_config.batchSize = 27
myloader:renew(myloader.l_config)
local batch = myloader:getBatch('train')
print(batch:size())
]]--

--local loader = paths.dofile('../data/data.lua')
--local dataset = loader.new(8, opt)
--local batch  = dataset:getBatch(45)
--print(batch:size())

--opt.batchSize = 24
--local dataset = loader.new(8, opt)
--local batch  = dataset:getBatch(45)
--print(batch:size())




-- create dataloader.
--local myloader = require 'script.myloader'



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
            ['num_channels']=3,
            ['resolution']=32,
            ['label_size']=0,
            ['fmap_base']=4096,
            ['fmap_decay']=1.0,
            ['fmap_max']=512,
            ['use_wscale']=true,
            ['fmap_gdrop']=true,
            ['fmap_layernorm']=false,
}

config = {
            ['D']=d_config,
            ['G']=g_config,
}

-- load players(gen, dis)
local gan_models = {}
local gan_gen = network.get_init_gen(g_config)
local gan_dis = network.get_init_dis(d_config)
--print ('Generator structure: ')    
--print(gan_gen)
--print ('Discriminator structure: ')    
--print(gan_dis)


gan_models = {gan_gen, gan_dis}


--[[
for i = 3, 4 do
    print('----------------')
    network.grow_network(gan_gen, gan_dis, i, g_config, d_config)
    --print(gan_dis)
end
]]--


--print(gan_gen.modules[2])
--local gan_dis = dis.create_model(opt.sampleSize, d_config)         -- discriminator
--local gan_gen = gen.create_model(g_config)         -- generator    
--gan_models = {gan_gen, gan_dis}
--print ('Generator structure: ')    
--print(gan_gen)

-- remove last layer.
--gan_gen:remove()
--print ('Generator structure after removal: ')    
--print(gan_gen)




--conv_nodes = gan_gen:findModules('nn.SpatialFullConvolution')
--for i = 1, #conv_nodes do
--    print(conv_nodes[i].__typename)
--end


--print ('Discriminator structure: ')
--print(gan_dis)

-- loss metrics
local gan_criterion = {nn.MSECriterion()}



-- run trainer
require 'script.pggan'
local optimstate = {}
local gan_trainer = PGGAN(gan_models, gan_criterion, opt, optimstate, config)
gan_trainer:train(10000, myloader)


print('Congrats! You just finished the training.')













  
