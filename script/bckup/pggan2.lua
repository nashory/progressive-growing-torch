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
require 'hdf5'
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
    self.batch_table = { [4]=32, [8]=32, [16]=32, [32]=16, [64]=16, [128]=16, [256]=12, [512]=4, [1024]=1 }         -- slightly different from the paper.
    self.batchSize = self.batch_table[math.pow(2, self.resl)]
    self.transition_tick = opt.transition_tick
    self.training_tick = opt.training_tick
    self.globalTick = 0
    self.fadein = {}
    self.complete= {['gen']=0.0, ['dis']=0.0}
    self.flag_flush_gen = true
    self.flag_flush_dis = true
    self.state = 'stab'


    -- init dataloader.
    self.loader = require 'script.myloader'
    self:renew_loader()

    
    -- generate test_noise(fixed)
    self.test_noise = torch.Tensor(self.batchSize, self.nz, 1, 1)
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

function PGGAN:feed_interpolated_input(src)
    local x = src
    if self.state == 'trns' and math.floor(self.resl) >2 then
        local x_intp = torch.Tensor(x:size()):zero()
        for i = 1, x_intp:size(1) do
            local x_up = x[{{i},{},{},{}}]:squeeze()
            local x_down = size_resample(size_resample(x_up, x:size(3)*0.5), x:size(3))
            x_intp[{{i},{},{},{}}]:copy(torch.add(x_up:mul(self.fadein.gen.alpha), x_down:mul(1.0-self.fadein.gen.alpha)))           -- iterpolated output
        end
        return x_intp
    else
        return x
    end

end


-- this function will schedule image resolution factor(resl) progressively.
-- should be called every iteration to ensure kimgs is counted properly.
-- step 1. (transition_tick) --> transition in both generator and discriminator.
-- step 2. (training_tick) --> stabilize.
-- total period: (training_tick + transition_tick)
function PGGAN:resl_scheduler()

    -- transition/training tick schedule.
    if math.floor(self.resl)==2 then
        self.training_tick = 100
        self.transition_tick = 100
    else
        self.training_tick = self.opt.training_tick
        self.transition_tick = self.opt.transition_tick
    end
    local delta = 1.0/(self.training_tick + self.transition_tick)
    local d_alpha = 1.0*self.batchSize/self.transition_tick/TICK
   
    -- update alpha if fade-in layer exist.
    if self.fadein.gen ~= nil and self.resl%1.0 <= (self.transition_tick)*delta then
        self.fadein.gen:updateAlpha(d_alpha)
        self.fadein.dis:updateAlpha(d_alpha)
        self.complete.gen = self.fadein.gen.complete
        self.complete.dis = self.fadein.dis.complete
        self.state = 'trns'
    end

    self.batchSize = self.batch_table[math.pow(2,math.floor(self.resl))]
    local prev_kimgs = self.kimgs
    self.kimgs = self.kimgs + self.batchSize
    if (self.kimgs%TICK) < (prev_kimgs%TICK) then
        self.globalTick = self.globalTick + 1
        -- increase linearly every tick, and grow network structure.
        local prev_resl = math.floor(self.resl)
        self.resl = self.resl + delta
        -- clamping, range: 4 ~ 1024
        self.resl = math.max(2, math.min(10.5, self.resl))

        -- flush network.
        if self.flag_flush_gen and self.flag_flush_dis and self.resl%1.0 >= (self.transition_tick)*delta then
            if self.fadein.gen ~= nil then
                self.fadein.gen:updateAlpha(d_alpha)
                self.complete.gen = self.fadein.gen.complete
            end
            if self.fadein.dis ~= nil then
                self.fadein.dis:updateAlpha(d_alpha)
                self.complete.dis = self.fadein.dis.complete
            end
            self.flag_flush_gen = false
            self.flag_flush_dis = false
            network.flush_FadeInBlock(self.gen, self.dis, math.ceil(self.resl), 'gen')
            network.flush_FadeInBlock(self.gen, self.dis, math.ceil(self.resl), 'dis')
            self.fadein.gen = nil
            self.fadein.dis = nil
            self.state = 'stab'
        end
        -- grow network
        if math.floor(self.resl) ~= prev_resl then
            self.flag_flush_gen = true
            self.flag_flush_dis = true
            self.batchSize = self.batch_table[math.pow(2, math.floor(self.resl))]
            network.grow_network(self.gen, self.dis, math.floor(self.resl),
                                 self.config.G, self.config.D, true)
            self:renew_loader()
            self:renew_parameters()
            -- reduce learning rate. (the authors did not use lr policy, but I will try.)
            --optimizer.dis.config.lr = optimizer.dis.config.lr*0.8
            --optimizer.gen.config.lr = optimizer.gen.config.lr*0.8

            -- find fadein layer.  
            local fadein_nodes = self.gen:findModules('nn.FadeInLayer')
            if #fadein_nodes~=0 then self.fadein.gen = fadein_nodes[1] end
            self.complete.gen = self.fadein.gen.complete
            local fadein_nodes = self.dis:findModules('nn.FadeInLayer')
            if #fadein_nodes~=0 then self.fadein.dis = fadein_nodes[1] end
            self.complete.dis = self.fadein.dis.complete
            self.state = 'trns'
        end
        if math.ceil(self.resl)>=11 and self.flag_flush then
            self.flag_flush_gen = false
            self.flag_flush_dis = false
            network.flush_FadeInBlock(self.gen, self.dis, math.ceil(self.resl), 'gen')
            network.flush_FadeInBlock(self.gen, self.dis, math.ceil(self.resl), 'dis')
            self.state = 'stab'
        end
    end
end


PGGAN['fDx'] = function(self, x)
    self.dis:zeroGradParameters()

    -- generate noise(z)
    self.noise = torch.Tensor(self.batch_table[math.pow(2, math.floor(self.resl))], self.nz, 1, 1):zero()
    self.label = torch.Tensor(self.batch_table[math.pow(2, math.floor(self.resl))], 1):zero()
    if self.noisetype == 'uniform' then self.noise:uniform(-1,1)
    elseif self.noisetype == 'normal' then self.noise:normal(0,1) end
    
    -- train with real(x)
    self.data = self.loader:getBatch('train')
    self.x = self:feed_interpolated_input(self.data:clone())
    self.label:fill(1)          -- real label (1)
    local fx = self.dis:forward(self.x:cuda())
    local d_errD_drift = -1*self.opt.epsilon_drift*(self.param_dis:clone():pow(2):sum()/self.param_dis:size(1))  -- get drift loss.
    local errD_real = self.crit_adv:forward(fx:cuda(), self.label:cuda())
    local d_errD_real = self.crit_adv:backward(fx:cuda(), self.label:cuda())
    local d_fx = self.dis:backward(self.x:cuda(), torch.add(d_errD_real, d_errD_drift):cuda())
    --local d_fx = self.dis:backward(self.x:cuda(), d_errD_real:cuda())

    -- train with fake(x_tilde)
    self.label:fill(0)          -- fake label (0)
    self.z = self.noise:clone()
    self.x_tilde = self.gen:forward(self.z:cuda()):clone()
    self.fx_tilde = self.dis:forward(self.x_tilde:cuda())
    local errD_fake = self.crit_adv:forward(self.fx_tilde:cuda(), self.label:cuda())
    local d_errD_fake = self.crit_adv:backward(self.fx_tilde:cuda(), self.label:cuda())
    local d_fx_tilde = self.dis:backward(self.x_tilde:cuda(), torch.add(d_errD_fake, d_errD_drift):cuda())
    --local d_fx_tilde = self.dis:backward(self.x_tilde:cuda(), d_errD_fake:cuda())
   
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
    optimizer.gen.optimstate = {}
    optimizer.dis.optimstate = {}
    self.param_gen, self.gradParam_gen = self.gen:getParameters()
    self.param_dis, self.gradParam_dis = self.dis:getParameters()
end

function PGGAN:renew_loader()
    self.loader.l_config.batchSize = self.batchSize
    self.loader.l_config.loadSize = math.pow(2, math.floor(self.resl))
    self.loader.l_config.sampleSize = math.pow(2, math.floor(self.resl))
    self.loader:renew(self.loader.l_config)
    
end

local stacked = 0
function PGGAN:train(loader)
    --self:test()
    --self:benchmark_GpuMemoryUsage()

    -- init logger
    os.execute('mkdir -p log')
    logger = optim.Logger(string.format('log/%s.log', self.opt.name))
    logger:setNames{'epoch', 'ticks', 'ErrD', 'ErrG', 'Res', 'Trn(G)', 'Trn(D)', 'Elp'}
    
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
        if stacked > math.ceil(self.loader:size()) then 
            epoch = epoch + 1 
            stacked = stacked%math.ceil(self.loader:size())
        end

        -- scheduling resolition transition
        self:resl_scheduler()
        
        -- forward / backward
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
        if (globalIter%self.opt.display_iter==0) and (self.opt.display) then
            local im_fake = self.gen:forward(self.test_noise:cuda()):clone()
            local im_fake_hq = size_resample(im_fake[{{1},{},{},{}}]:squeeze(), 1024)               -- hightest resolution we are targeting.
            local im_real_hq = size_resample(self.x[{{1},{},{},{}}]:squeeze(), 1024)                -- hightest resolution we are targeting.
            local grid_test = create_img_grid(im_fake:clone(), 128, 4)                  -- 4x4 grid.
            local grid_rand = create_img_grid(self.x_tilde:clone(), 128, 4)             -- 4x4 grid.

                
            self.disp.image(grid_test, {win=self.opt.display_id*1 + self.opt.gpuid, title=self.opt.server_name})
            self.disp.image(grid_rand, {win=self.opt.display_id*2 + self.opt.gpuid, title=self.opt.server_name})
            self.disp.image(im_fake_hq, {win=self.opt.display_id*4 + self.opt.gpuid, title=self.opt.server_name})
            self.disp.image(im_real_hq, {win=self.opt.display_id*8 + self.opt.gpuid, title=self.opt.server_name})
            if (globalIter%(self.opt.display_iter*self.opt.save_jpg_iter)==0) then
                -- save image as jpg grid.
                os.execute(string.format('mkdir -p save/grid'))   
                image.save( string.format('save/grid/%d_%s_%.1f_%.1f.jpg', 
                            math.floor(globalIter/self.opt.display_iter), 
                            self.state, self.complete.gen, self.complete.dis), 
                            grid_test)
                -- save generated HQ fake image.            
                os.execute(string.format('mkdir -p save/resl_%d', math.pow(2, math.floor(self.resl))))
                image.save( string.format('save/resl_%d/%d.jpg', 
                            math.pow(2, math.floor(self.resl)), 
                            math.floor(globalIter/self.opt.display_iter)), 
                            im_fake_hq:add(1):div(2))
            end
        end

        -- snapshot (save model)
        local data = {  ['dis'] = self.dis:clearState(), 
                        ['gen'] = self.gen:clearState(),
                        ['state'] = {   ['dis'] = optimizer.dis.optimstate,
                                        ['gen'] = optimizer.gen.optimstate,
                                        ['resl'] = self.resl,}}
        self:snapshot(string.format('repo/%s', self.opt.name), self.opt.name, data)

        -- logging
        local log_msg = string.format('[E:%d][T:%d][%6d/%6d]  errD(real): %.4f | errD(fake): %.4f | errG: %.4f  [lr:%.3fe-3][Res:%4d][Trn(G):%.1f%%][Trn(D):%.1f%%][Elp(hr):%.4f]',
                                        epoch, self.globalTick, stacked, self.loader:size(), 
                                        errD.real, errD.fake, errG, optimizer.gen.config.lr*1000, math.pow(2,math.floor(self.resl)), 
                                        self.complete.gen, self.complete.dis, tm:time().real/3600.0)
        print(log_msg)
        logger:setNames{'epoch', 'ticks', 'ErrD', 'ErrG', 'Res', 'Trn(G)', 'Trn(D)', 'Elp'}
        logger:add({epoch, self.globalTick, errD.real+errD.fake, errG, 
                    math.pow(2,math.floor(self.resl), self.complete.gen, 
                    self.complete.dis, tm:time().real/3600.0)})
    

    end
    -- stop timer.
    tm:stop()
end


function PGGAN:snapshot(path, fname, data)
    -- if dir not exist, create it.
    if not paths.dirp(path) then    os.execute(string.format('mkdir -p %s', path)) end
    -- save every 100 tick if the network is in stabilization phase. 
    local fname = fname ..'_R' .. math.floor(self.resl) .. '_T' .. self.globalTick
    if self.globalTick%100==0 and self.state=='stab' then
        local save_path = path .. '/' .. fname
        if not paths.filep(save_path .. '_dis.t7') then torch.save(save_path .. '_dis.t7', data.dis) end
        if not paths.filep(save_path .. '_gen.t7') then torch.save(save_path .. '_gen.t7', data.dis) end
        if not paths.filep(save_path .. '_state.t7') then 
            torch.save(save_path .. '_state.t7', data.dis)
            print('[Snapshot]: saved model @ ' .. save_path)
        end
    end
end



----------------------- Debugging functions --------------------------
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

function PGGAN:benchmark_GpuMemoryUsage()
    require 'cutorch'
    self.batchSize = self.batch_table[math.pow(2, math.floor(self.resl))]
    for resl = 2, 11 do
        
        print('---------------------------------------------------------------------------------------')
        local free, total = cutorch.getMemoryUsage(self.opt.gpuid+1)
        print(string.format('[%s] resl:%d, total:%d MB, free:%d MB', 'before flush', math.floor(self.resl), total/1024/1024, free/1024/1024))
        network.flush_FadeInBlock(self.gen, self.dis, resl)
        local free, total = cutorch.getMemoryUsage(self.opt.gpuid+1)
        print(string.format('[%s] resl:%d, total:%d MB, free:%d MB', 'after flush', math.floor(self.resl), total/1024/1024, free/1024/1024))
        if resl ~= 11 then
            self.resl = resl
            network.grow_network(self.gen, self.dis, resl,
                                    self.config.G, self.config.D, true)
            self.batchSize = self.batch_table[math.pow(2, math.floor(self.resl))]
            local free, total = cutorch.getMemoryUsage(self.opt.gpuid+1)
            print(string.format('[%s] resl:%d, total:%d MB, free:%d MB', 'after growing', math.floor(self.resl), total/1024/1024, free/1024/1024))
            self:renew_loader()
            local errD = self:fDx()
            local errG = self:fGx()
            -- find fadein layer.  
            fadein_nodes = self.gen:findModules('nn.FadeInLayer')
            if #fadein_nodes~=0 then self.fadein = fadein_nodes[1] end
        end

    end
end


return PGGAN




