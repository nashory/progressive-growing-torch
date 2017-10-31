-- For training loop and learning rate scheduling.
-- Progressive-growing GAN from NVIDIA.
-- last modified : 2017.10.30, nashory

-- 1 cycle = 1000 ticks
-- we do this since the batchsize varies depending on the image resolution.




require 'sys'
require 'optim'
require 'image'
require 'math'
local optimizer = require 'script.optimizer'


local PGGAN = torch.class('PGGAN')


function PGGAN:__init(model, criterion, opt, optimstate)
    self.model = model
    self.criterion = criterion
    self.optimstate = optimstate or {
        lr = opt.lr,
    }
    self.opt = opt
    self.noisetype = opt.noisetype
    self.nc = opt.nc
    self.nh = opt.nh
    self.gamma = opt.gamma
    self.lambda = opt.lambda
    self.kt = 0         -- initialize same with the paper.
    self.batchSize = opt.batchSize
    self.sampleSize = opt.sampleSize
    self.thres = 1.0

    self.batch_table = { [4]=50, [8]=50, [16]=50, [32]=20, [64]=10, [128]=10, [256]=5, [512]=2, [1024]=1 }

    
    -- generate test_noise(fixed)
    self.test_noise = torch.Tensor(64, self.nh)
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

function PGGAN:ResolutionScheduler()
    print('this function will schedule image resolution factor progressively.')
end

function PGGAN:UpdateVarsExternally()
    print('update layer variables externally (e.g. cur_lod in FadeInLayer)')
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


return PGGAN




