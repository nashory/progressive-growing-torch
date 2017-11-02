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
    self.gamma = opt.gamma
    self.lambda = opt.lambda
    self.kt = 0         -- initialize same with the paper.
    self.batchSize = opt.batchSize
    self.sampleSize = opt.sampleSize
    self.thres = 1.0

    -- init dataloader.
    self.loader = require 'script.myloader'
    self.loader:renew(self.loader.l_config)

    -- initial variables.
    self.config = config
    self.resl = 2                   -- range from [2, 10] --> [4, 1024]
    self.kimgs = 0                  -- accumulated total number of images forwarded.
    self.batch_table = { [4]=64, [8]=64, [16]=32, [32]=32, [64]=16, [128]=16, [256]=12, [512]=4, [1024]=1 }         -- slightly different from the paper.
    self.transition_tick = opt.transition_tick
    self.training_tick = opt.training_tick
    self.flag_flush = true
    self.complete = 100.0
    self.globalTick = 0


    
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

    local batchSize = self.batch_table[math.pow(2,math.floor(self.resl))]
    local prev_kimgs = self.kimgs
    self.kimgs = self.kimgs + batchSize
    if (self.kimgs%TICK) < (prev_kimgs%TICK) then
        self.globalTick = self.globalTick + 1
        -- increase linearly every tick, and grow network structure.
        local prev_resl = math.floor(self.resl)
        self.resl = self.resl + 1.0/(self.transition_tick + self.training_tick)
        -- clamping, range: 4 ~ 1024
        self.resl = math.max(2, math.min(10, self.resl))

        -- remove fade-in block
        if self.resl%1.0 > (1.0*self.transition_tick)/(self.transition_tick+self.training_tick) then
            if self.flag_flush then
                network.flush_FadeInBlock(self.gen, self.dis, math.floor(self.resl))
                self.flag_flush = false
            end
        end
        -- grow network
        if math.floor(self.resl) ~= prev_resl then
            self.flag_flush = true
            network.grow_network(self.gen, self.dis, math.floor(self.resl),
                                 self.config.G, self.config.D, true)
        end
    end
    fadein_nodes = self.gen:findModules('nn.FadeInLayer')
    if #fadein_nodes~=0 then
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
    -- generate noise(z_D)
    self.noise = torch.Tensor(self.batch_table[math.pow(2, math.floor(self.resl))], self.nz, 1, 1)
    if self.noisetype == 'uniform' then self.noise:uniform(-1,1)
    elseif self.noisetype == 'normal' then self.noise:normal(0,1) end
    
    local x_tilde = self.gen:forward(self.noise:cuda())
    local predict = self.dis:forward(x_tilde:cuda())
end

PGGAN['fGx'] = function(self, x)
    print('fDx')
end



function PGGAN:train(epoch, loader)
    -- get network weights.
    print(string.format('Dataset size :  %d', self.loader:size()))
    self.gen:training()
    self.dis:training()
    self.param_gen, self.gradParam_gen = self.gen:getParameters()
    self.param_dis, self.gradParam_dis = self.dis:getParameters()

    local globalIter = 0
    -- calculate iteration.
    local iter_per_epoch = math.ceil(self.loader:size()/self.batchSize)
    for e = 1, epoch do
        for iter = 1, iter_per_epoch do
            globalIter = globalIter + 1
            -- grow network automatically.
            self:ResolutionScheduler()
            self:fDx()

            print(string.format('Iter: %d | Tick:%d | Tranition: %.2f', globalIter, self.globalTick, self.complete))
        end
    end
end

            
            
            
            --self:test()
            --[[
            -- forward/backward and update weights with optimizer.
            -- DO NOT CHANGE OPTIMIZATION ORDER.
            local err_dis = self:fDx()
            local err_gen = self:fGx()

            -- weight update.
            optimizer.dis.method(self.param_dis, self.gradParam_dis, optimizer.dis.config.lr,
                                optimizer.dis.config.beta1, optimizer.dis.config.beta2,
                                optimizer.dis.config.elipson, optimizer.dis.optimstate)
            optimizer.gen.method(self.param_gen, self.gradParam_gen, optimizer.gen.config.lr,
                                optimizer.gen.config.beta1, optimizer.gen.config.beta2,
                                optimizer.gen.config.elipson, optimizer.gen.optimstate)

            -- save model at every specified epoch.
            --local data = {dis = self.dis, gen = self.gen}
            --self:snapshot(string.format('repo/%s', self.opt.name), self.opt.name, totalIter, data)

            -- logging
            --local log_msg = string.format('Epoch: [%d][%6d/%6d]  D(real): %.4f | D(fake): %.4f | G: %.4f | Delta: %.4f | kt: %.6f | Convergence: %.4f', e, iter, iter_per_epoch, err_dis.real, err_dis.fake, err_gen.err, err_gen.delta, self.kt, self.measure)
            --print(log_msg)
            ]]--


--[[

function PGGAN:getSamples(batchSize, resolution)
    --local batch = self.dataset:sample()
    local batch = self.dataset:getBatch()
end

function PGGAN:ResolutionScheduler()
    print('this function will schedule image resolution factor progressively.')
end

-- update layer variables externally (e.g. cur_lod in FadeInLayer)
function PGGAN:UpdateVarsExternally(model)
    local targ_node = model:findModules('nn.FadeInLayer')
    for i=1, #target_node do
        print(targ_node[i].__typename)
        --targ_node[i].cur_lod = xx
    end
end



PGGAN['fDx'] = function(self, x)
    self.dis:zeroGradParameters()
    
    -- generate noise(z_D)
    if self.noisetype == 'uniform' then self.noise:uniform(-1,1)
    elseif self.noisetype == 'normal' then self.noise:normal(0,1) end
   
    -- forward with real(x) and fake(x_tilde)
    self.x = self.dataset:getBatch()
    self.z = self.noise:clone():cuda()
    self.gen:forward(self.z)
    self.x_tilde = self.gen.output:clone()
    
    self.dis:forward({self.x:cuda(), self.x_tilde:cuda()})
    self.ae = self.dis.output
    
    -- loss calculation
    self.errD_real = self.crit_adv:forward(self.ae[1]:cuda(), self.x:cuda())
    local d_errD_real = self.crit_adv:backward(self.ae[1]:cuda(), self.x:cuda()):clone()
    self.errD_fake = self.crit_adv:forward(self.ae[2]:cuda(), self.x_tilde:cuda())
    local d_errD_fake = self.crit_adv:backward(self.ae[2]:cuda(), self.x_tilde:cuda()):clone()
    
    -- backward discriminator
    local d_ae = self.dis:backward({self.x:cuda(), self.x_tilde:cuda()}, {d_errD_real:cuda(), d_errD_fake:mul(-self.kt):cuda()})
    
    -- return error.
    local errD = {real = self.errD_real, fake = self.errD_fake}
    return errD
end


PGGAN['fGx'] = function(self, x)
    self.gen:zeroGradParameters()
   
    -- generate noise(z_G)
    if self.noisetype == 'uniform' then self.noise:uniform(-1,1)
    elseif self.noisetype == 'normal' then self.noise:normal(0,1) end
    
    local x_ae = self.dis.output
    local errG = self.crit_adv:forward(x_ae[2]:cuda(), self.x_tilde:cuda())
    
    local d_errG = self.crit_adv:updateGradInput(x_ae[2]:cuda(), self.x_tilde:cuda()):clone()
    local d_gen_dis = self.dis:updateGradInput({self.x:cuda(), self.x_tilde:cuda()}, {d_errG:zero():cuda(), d_errG:cuda()})
    local d_gen_dummy = self.gen:backward(self.z:cuda(), d_gen_dis[2]:cuda()):clone()

    -- closed loop control for kt
    local delta = self.gamma*self.errD_real - errG
    self.kt = self.kt + self.lambda*(delta)

    if self.kt > self.thres then self.kt = self.thres
    elseif self.kt < 0 then self.kt = 0 end

    -- Convergence measure
    self.measure = self.errD_real + math.abs(delta)

    return {err = errG, delta = delta}
end


function PGGAN:train(epoch, loader)
    -- Initialize data variables.
    self.noise = torch.Tensor(self.batchSize, self.nh)

    -- get network weights.
    self.dataset = loader.new(self.opt.nthreads, self.opt)
    print(string.format('Dataset size :  %d', self.dataset:size()))
    self.gen:training()
    self.dis:training()
    self.param_gen, self.gradParam_gen = self.gen:getParameters()
    self.param_dis, self.gradParam_dis = self.dis:getParameters()


    local totalIter = 0
    for e = 1, epoch do
        -- get max iteration for 1 epcoh.
        local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
        for iter  = 1, iter_per_epoch do
            totalIter = totalIter + 1

            -- forward/backward and update weights with optimizer.
            -- DO NOT CHANGE OPTIMIZATION ORDER.
            local err_dis = self:fDx()
            local err_gen = self:fGx()

            -- weight update.
            optimizer.dis.method(self.param_dis, self.gradParam_dis, optimizer.dis.config.lr,
                                optimizer.dis.config.beta1, optimizer.dis.config.beta2,
                                optimizer.dis.config.elipson, optimizer.dis.optimstate)
            optimizer.gen.method(self.param_gen, self.gradParam_gen, optimizer.gen.config.lr,
                                optimizer.gen.config.beta1, optimizer.gen.config.beta2,
                                optimizer.gen.config.elipson, optimizer.gen.optimstate)

            -- save model at every specified epoch.
            local data = {dis = self.dis, gen = self.gen}
            self:snapshot(string.format('repo/%s', self.opt.name), self.opt.name, totalIter, data)

            -- display server.
            if (totalIter%self.opt.display_iter==0) and (self.opt.display) then
                local im_real = self.x:clone()
                local im_fake = self.x_tilde:clone()
                local im_real_ae = self.ae[1]:clone()
                local im_fake_ae = self.ae[2]:clone()
                
                self.disp.image(im_real, {win=self.opt.display_id + self.opt.gpuid, title=self.opt.server_name})
                self.disp.image(im_fake, {win=self.opt.display_id*2 + self.opt.gpuid, title=self.opt.server_name})
                self.disp.image(im_real_ae, {win=self.opt.display_id*4 + self.opt.gpuid, title=self.opt.server_name})
                self.disp.image(im_fake_ae, {win=self.opt.display_id*6 + self.opt.gpuid, title=self.opt.server_name})


                -- save image as png (size 64x64, grid 8x8 fixed).
                local im_png = torch.Tensor(3, self.sampleSize*8, self.sampleSize*8):zero()
                self.gen:forward(self.test_noise:cuda())
                local x_test = self.gen.output:clone()
                for i = 1, 8 do
                    for j =  1, 8 do
                        im_png[{{},{self.sampleSize*(j-1)+1, self.sampleSize*(j)},{self.sampleSize*(i-1)+1, self.sampleSize*(i)}}]:copy(x_test[{{8*(i-1)+j},{},{},{}}]:clone():add(1):div(2))
                    end
                end
                os.execute('mkdir -p repo/image')
                image.save(string.format('repo/image/%d.jpg', totalIter/self.opt.display_iter), im_png)
            end

            -- logging
            local log_msg = string.format('Epoch: [%d][%6d/%6d]  D(real): %.4f | D(fake): %.4f | G: %.4f | Delta: %.4f | kt: %.6f | Convergence: %.4f', e, iter, iter_per_epoch, err_dis.real, err_dis.fake, err_gen.err, err_gen.delta, self.kt, self.measure)
            print(log_msg)
        end
    end
end


function PGGAN:snapshot(path, fname, iter, data)
    -- if dir not exist, create it.
    if not paths.dirp(path) then    os.execute(string.format('mkdir -p %s', path)) end
    
    local fname = fname .. '_Iter' .. iter .. '.t7'
    local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
    if iter % math.ceil(self.opt.snapshot_every*iter_per_epoch) == 0 then
        local save_path = path .. '/' .. fname
        torch.save(save_path)
        print('[Snapshot]: saved model @ ' .. save_path)
    end
end
]]--


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




