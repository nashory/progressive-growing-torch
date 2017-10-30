-- Discriminator network structure.
require 'custom_layers'



local nn = require 'nn'


local Dicrim = {}

local SBatchNorm = nn.SpatialBatchNormalization
local SConv = nn.SpatialConvolution
local AvgPool = nn.SpatialAveragePooling
local Minibatch = nn.MinibatchStatConcat


function Discrim.weights_init(m)
	local name = torch.type(m)
	if name:find('Convolution') then
		m.weight:normal(0.0, 0.02)
		m.bias:fill(0)
	elseif name:find('BarchNormalization') then
		if m.weight then m.weight:normal(1.0, 0.02) end
		if m.bais then m.bais:fill(0) end
	end
end

-- create generator structure.
function Discrim.create_model(config)
   
   
    local model = nn.Sequential()

    local nz = config['nz']
    local ndf = config['fmap_max']
    local nchannel = config['num_channels']
    
    flag_bn = config['use_bathnorm']
    flag_lrelu = config['use_leakyrelu']
    flag_pxlnorm = config['use_pixelnorm']
    flag_tanh = config['use_tanh']
    
    -- set initial block.
    local input_block = nn.Sequential()
    input_block:add(SConv(nchannel, ndf/32, 1, 1))
    if flag_bn then input_block:add(SBatchNorm(ngf)) end
    if flag_lrelu then input_block:add(nn.LeakyReLU(0.2,true)) else input_block:add(nn.ReLU(true)) end
    input_block:add(SConv(ndf/32, ndf/32, 3, 3, 1, 1, 1, 1))
    if flag_bn then input_block:add(SBatchNorm(ngf)) end
    if flag_lrelu then input_block:add(nn.LeakyReLU(0.2,true)) else input_block:add(nn.ReLU(true)) end
    input_block:add(SConv(ndf/32, ndf/16, 3, 3, 1, 1, 1, 1))
    if flag_bn then input_block:add(SBatchNorm(ngf)) end
    if flag_lrelu then input_block:add(nn.LeakyReLU(0.2,true)) else input_block:add(nn.ReLU(true)) end
    input_block:add(AvgPool(2,2,2,2))           -- downsample by factor of 2
    -- state size : ngf x 4 x 4
    model:add(input_block)

    -- intermediate blocks.
    local ndim = nil
    for i = 1, 7 do
        ndim = ndf
        if i == 1 then
            for k = 1, (5-i) do ndim = ndim/2 end
            local inter_block = Generator.intermediate_block(ndim, true)
            model:add(inter_block)
        elseif i==2 or i==3 or i==4 then
            for k = 1, (5-i) do ndim = ndim/2 end
            local inter_block = Generator.intermediate_block(ndim, true)
            model:add(inter_block)
        elseif i==5 or i==6 or i==7 then
            local inter_block = Generator.intermediate_block(ndim, false)
            model:add(inter_block)
        end
    end

    -- final block.
    local output_block = nn.Sequential()
    output_block:add(Minibatch(ndim))
    -- conv1 (3x3)
    output_block:add(SConv(ndim, ndim, 3, 3, 1, 1, 1, 1))
    if flag_bn then output_block:add(SBatchNorm(ndim/2)) end
    if flag_lrelu then output_block:add(nn.LeakyReLU(0.2,true)) else output_block:add(nn.ReLU(true)) end
    -- conv2 (3x3)
    output_block:add(SConv(ndim, ndim, 4, 4))
    if flag_bn then output_block:add(SBatchNorm(ndim/2)) end
    if flag_lrelu then output_block:add(nn.LeakyReLU(0.2,true)) else output_block:add(nn.ReLU(true)) end
    -- Linear (1x1)
    output_block:add(View(-1, 1))
    output_block:add(nn.Linear(ndim, 1))
    output_block:add(nn.Sigmoid())
    model:add(output_block)

    return model
end

function Discrim.intermediate_block(ndim, doubling)
    doubling = doubling or false
    local inter_block = nn.Sequential()
    
    
    if doubling then ndf = ndim*2 else ndf = ndim end
    -- conv1
    inter_block:add(SConv(ndim, ndf, 3, 3, 1, 1, 1, 1))
    if flag_bn then inter_block:add(SBatchNorm(ndf)) end
    if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
    -- conv2
    inter_block:add(SConv(ndf, ndf, 3, 3, 1, 1, 1, 1))
    if flag_bn then inter_block:add(SBatchNorm(ndf)) end
    if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
    inter_block:add(AvgPool(2,2,2,2))           -- scale down by factor of 2.0

    return inter_block
end

return Generator



