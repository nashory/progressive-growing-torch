-- For training loop and learning rate scheduling.
-- Progressive-growing GAN from NVIDIA.
-- last modified : 2017.10.30, nashory

-- 1 cycle = 1 tick x (transition_tick + training_tick)
-- 1 tick = 1K images = (1K x batchsize) iters
-- we do this since the batchsize varies depending on the image resolution.

-- Notation:
--      resl: resolution level
--      x: real image
--      x_tilde: fake image




require 'sys'
require 'optim'
require 'image'
require 'math'
require 'script.utils'
local network = require 'models.network'
local optimizer = require 'script.optimizer'


local PGGAN = torch.class('PGGAN')
TICK = 1000

function PGGAN:__init(model, criterion, opt, optimstate, config)
    self.model = model
    self.criterion = criterion
    self.optimstate = optimstate or {
        lr = opt.lr,
    }
    self.opt = opt
    self.noisetype = opt.noisetype
    self.nc = opt.nc
    self.nz = opt.nz


    -- initial variables.
    self.config = config
    self.resl = 2                   -- range from [2, 10] --> [4, 1024]
    self.kimgs = 0                  -- accumulated total number of images forwarded.
    self.batch_table = { [4]=64, [8]=64, [16]=32, [32]=32, [64]=16, [128]=16, [256]=12, [512]=4, [1024]=1 }         -- slightly different from the paper.
    self.batchSize = self.batch_table[math.pow(2, self.resl+1)]
    self.transition_tick = opt.transition_tick
    self.training_tick = opt.training_tick
    self.flag_flush = true
    self.complete = 100.0
    self.globalTick = 0

    -- init dataloader.
    self.loader = require 'script.myloader'
    self:renew_loader()

    
    -- generate test_noise(fixed)
    self.test_noise = torch.Tensor(64, self.nz, 1, 1)
    if self.noisetype == 'uniform' then self.test_noise:uniform(-1,1)
    elseif self.noisetype == 'normal' then self.test_noise:normal(0,1) end
    
    if opt.display then
        self.disp = require 'display'
        self.disp.configure({hostname=opt.display_server_ip, port=opt.display_server_port})
    end

    -- get models and criterion.
    self.gen = model[1]:cuda()
    self.dis = model[2]:cuda()
    self.crit_adv = criterion[1]:cuda()
end


-- this function will schedule image resolution factor(resl) progressively.
-- should be called every iteration to ensure kimgs is counted properly.
function PGGAN:ResolutionScheduler()

    self.batchSize = self.batch_table[math.pow(2,math.floor(self.resl+1))]
    local prev_kimgs = self.kimgs
    self.kimgs = self.kimgs + self.batchSize
    if (self.kimgs%TICK) < (prev_kimgs%TICK) then
        self.globalTick = self.globalTick + 1
        -- increase linearly every tick, and grow network structure.
        local prev_resl = math.floor(self.resl)
        self.resl = self.resl + 1.0/(self.transition_tick + self.training_tick)
        -- clamping, range: 4 ~ 1024
        self.resl = math.max(2, math.min(9, self.resl))

        -- remove fade-in block
        if self.resl%1.0 > (1.0*self.transition_tick)/(self.transition_tick+self.training_tick) then
            if self.flag_flush then
                network.flush_FadeInBlock(self.gen, self.dis, math.floor(self.resl))
                self.flag_flush = false
                self:renew_parameters()
            end
        end
        -- grow network
        if math.floor(self.resl) ~= prev_resl then
            self.flag_flush = true
            self.batchSize = self.batch_table[math.pow(2, math.floor(self.resl+1))]
            network.grow_network(self.gen, self.dis, math.floor(self.resl),
                                 self.config.G, self.config.D, true)
            self:renew_loader()
            self:renew_parameters()
        end
    end

    fadein_nodes = self.gen:findModules('nn.FadeInLayer')
    if #fadein_nodes~=0 then
        fadein_nodes[1]:updateAlpha(self.batchSize)
        self.complete = fadein_nodes[1].complete
    end
end

function PGGAN:test()
    -- generate noise(z_D)
    self.noise = torch.Tensor(self.batch_table[math.pow(2, self.resl)], self.nz, 1, 1)
    if self.noisetype == 'uniform' then self.noise:uniform(-1,1)
    elseif self.noisetype == 'normal' then self.noise:normal(0,1) end
    
    local x_tilde = self.gen:forward(self.noise:cuda())
    print(x_tilde:size())
    local predict = self.dis:forward(x_tilde:cuda())
    print(predict:size())
end


PGGAN['fDx'] = function(self, x)
    self.dis:zeroGradParameters()

    -- generate noise(z)
    self.noise = torch.Tensor(self.batch_table[math.pow(2, math.floor(self.resl+1))], self.nz, 1, 1):zero()
    self.label = torch.Tensor(self.batch_table[math.pow(2, math.floor(self.resl+1))], 1):zero()
    if self.noisetype == 'uniform' then self.noise:uniform(-1,1)
    elseif self.noisetype == 'normal' then self.noise:normal(0,1) end
    
    -- train with real(x)
    self.x = self.loader:getBatch('train'):clone()
    self.label:fill(1)          -- real label (1)
    local fx = self.dis:forward(self.x:cuda())
    local errD_real = self.crit_adv:forward(fx:cuda(), self.label:cuda())
    local d_errD_real = self.crit_adv:backward(fx:cuda(), self.label:cuda())
    local d_fx = self.dis:backward(self.x:cuda(), d_errD_real:cuda())

    -- train with fake(x_tilde)
    self.label:fill(0)          -- fake label (0)
    self.z = self.noise:clone()
    self.x_tilde = self.gen:forward(self.z:cuda()):clone()
    self.fx_tilde = self.dis:forward(self.x_tilde:cuda())
    local errD_fake = self.crit_adv:forward(self.fx_tilde:cuda(), self.label:cuda())
    local d_errD_fake = self.crit_adv:backward(self.fx_tilde:cuda(), self.label:cuda())
    local d_fx_tilde = self.dis:backward(self.x_tilde:cuda(), d_errD_fake:cuda())
   
    -- return error.
    local errD = {  ['real'] = errD_real,
                    ['fake'] = errD_fake}
    return errD
end

PGGAN['fGx'] = function(self, x)
    self.gen:zeroGradParameters()
    self.label:fill(1)          -- fake label (1) --> reparameterization trick.
    local errG = self.crit_adv:forward(self.fx_tilde:cuda(), self.label:cuda())
    local d_errG = self.crit_adv:backward(self.fx_tilde:cuda(), self.label:cuda())
    local d_fx_tilde = self.dis:updateGradInput(self.x_tilde:cuda(), d_errG:cuda())
    local d_gen_dummy = self.gen:backward(self.z:cuda(), d_fx_tilde:cuda())

    return errG
end


function PGGAN:renew_parameters()
    self.gen:training()
    self.dis:training()
    self.param_gen = nil
    self.param_dis = nil
    optimizer.gen.optimstate = {}
    optimizer.dis.optimstate = {}
    self.param_gen, self.gradParam_gen = self.gen:getParameters()
    self.param_dis, self.gradParam_dis = self.dis:getParameters()
end

function PGGAN:renew_loader()
    self.loader.l_config.batchSize = self.batchSize
    self.loader.l_config.loadSize = math.pow(2, math.floor(self.resl+1))
    self.loader.l_config.sampleSize = math.pow(2, math.floor(self.resl+1))
    self.loader:renew(self.loader.l_config)
    
end

local stacked = 0
function PGGAN:train(loader)
    -- get network weights.
    print(string.format('Dataset size :  %d', self.loader:size()))
    
    -- get parameters from networks.
    self:renew_parameters()

    -- let's do training!
    local globalIter = 0
    local epoch = 0
    local tm = torch.Timer()
    for iter = 1, TICK*self.opt.total_tick do
        -- calculate iteration.
        globalIter = globalIter + 1
        stacked = stacked + self.batchSize
        if stacked%(math.ceil(self.loader:size()))==0 then 
            epoch = epoch + 1 
            stacked = stacked%meth.ceil(self.loader:size())
        end
        
        self:ResolutionScheduler()
        local errD = self:fDx()
        local errG = self:fGx()
           
        -- weight update.
        optimizer.dis.method(self.param_dis, self.gradParam_dis, optimizer.dis.config.lr,
                             optimizer.dis.config.beta1, optimizer.dis.config.beta2,
                             optimizer.dis.config.elipson, optimizer.dis.optimstate)
        optimizer.gen.method(self.param_gen, self.gradParam_gen, optimizer.gen.config.lr,
                             optimizer.gen.config.beta1, optimizer.gen.config.beta2,
                             optimizer.gen.config.elipson, optimizer.gen.optimstate)

        -- display
        -- 


            
        -- logging
        local log_msg = string.format('[E:%d][T:%d][%6d/%6d]    errD(real): %.4f | errD(fake): %.4f | errG: %.4f    [Res:%4d][Trn:%.1f%%][Elp(hr):%.4f]',
                                        epoch, self.globalTick, stacked, self.loader:size(), 
                                        errD.real, errD.fake, errG, math.pow(2,math.floor(self.resl+1)), self.complete, tm:time().real/3600.0)
        print(log_msg)
        
    end
    -- stop timer.
    tm:stop()
end



----------------------------------- Debugging functions --------------------------------------


function PGGAN:__debug__output()
    print(self.dis)
    self.resl = self.resl + 1
    self:test()
    network.grow_network(self.gen, self.dis, self.resl, self.config.G, self.config.D, true)
    print(self.dis)
    self.resl = self.resl + 1
    self:test() 
    network.grow_network(self.gen, self.dis, self.resl, self.config.G, self.config.D, true)
    print(self.dis)
    self.resl = self.resl + 1
    self:test() 
    network.grow_network(self.gen, self.dis, self.resl, self.config.G, self.config.D, true)
    print(self.dis)
    self.resl = self.resl + 1
    self:test() 
    network.grow_network(self.gen, self.dis, self.resl, self.config.G, self.config.D, true)
    print(self.dis)
    self.resl = self.resl + 1
    self:test() 
    network.grow_network(self.gen, self.dis, self.resl, self.config.G, self.config.D, true)
    print(self.dis)
    self.resl = self.resl + 1
    self:test() 
    network.grow_network(self.gen, self.dis, self.resl, self.config.G, self.config.D, true)
    print(self.dis)
    self.resl = self.resl + 1
    self:test() 
    print('-----')
    network.grow_network(self.gen, self.dis, self.resl, self.config.G, self.config.D, true)
    print(self.dis)
    self.resl = self.resl + 1
    self:test() 
end

return PGGAN




