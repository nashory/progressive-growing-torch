-- dataloader
-- this scripy is specifically designed for progressive-growing GAN.
-- will change this to multi-threaded 'loader' module for speed up.



local opts = require 'script.opts'
local loader = require 'data.loader'
local opt = opts.parse(arg)


local myloader = {}

-- initial setting for dataloader.
myloader.l_config = {}
myloader.l_config.trainPath = opt.data_root_train
myloader.l_config.nthreads = opt.nthreads
myloader.l_config.batchSize = 64
myloader.l_config.loadSize = 8
myloader.l_config.sampleSize = 8
myloader.l_config.rotate = 0
myloader.l_config.crop = 'none'
myloader.l_config.padding = true
myloader.l_config.hflip = false
myloader.l_config.keep_ratio = true
myloader.l_config.whitenoise = 0
myloader.l_config.brightness = 0
print(myloader.l_config)

-- create dataloader.
local dataset = loader.new(myloader.l_config)


function myloader.renew(self, l_config)
    print('re-configure dataloader setting...')
    dataset = loader.new(l_config)
end

function myloader:getBatch(target)
    return dataset:getBatch(target)
end

function myloader:size()
    return dataset:size()
end

return myloader






