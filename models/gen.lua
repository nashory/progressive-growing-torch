-- Generator network structure.

local nn = require 'nn'


local Generator = {}

local SBatchNorm = nn.SpatialBatchNormalization
local SConv = nn.SpatialConvolution
local SFullConv = nn.SpatialFullConvolution
local UpSampleNearest = nn.SpatialUpSamplingNearest


function Generator.weights_init(m)
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
function Generator.create_model(g_config)
   
   
    local model = nn.Sequential()

    local nz = g_config['nz']
    local ngf = g_config['fmap_max']
    local nchannel = g_config['num_channels']
    
    flag_bn = g_config['use_bathnorm']
    flag_lrelu = g_config['use_leakyrelu']
    flag_pxlnorm = g_config['use_pixelnorm']
    flag_tanh = g_config['use_tanh']
    
    -- set initial block.
    local input_block = nn.Sequential()
    input_block:add(SFullConv(nz, ngf, 4, 4, 2, 2, 1, 1):noBias())
    if flag_bn then input_block:add(SBatchNorm(ngf)) end
    if flag_lrelu then input_block:add(nn.LeakyReLU(0.2,true)) else input_block:add(nn.ReLU(true)) end
    input_block:add(SFullConv(nz, ngf, 3, 3, 1, 1, 1, 1):noBias())
    if flag_bn then input_block:add(SBatchNorm(ngf)) end
    if flag_lrelu then input_block:add(nn.LeakyReLU(0.2,true)) else input_block:add(nn.ReLU(true)) end
    -- state size : ngf x 4 x 4
    model:add(input_block)

    -- intermediate blocks.
    local ndim = nil
    for i = 1, 7 do
        ndim = ngf
        if i == 1 then
            local inter_block = Generator.intermediate_block(ndim, false)
            model:add(inter_block)
        elseif i==2 or i==3 then
            local inter_block = Generator.intermediate_block(ndim, false)
            model:add(inter_block)
        elseif i==4 or i==5 or i==6 or i==7 then
            for k = 1, (i-3) do ndim = ndim/2 end
            local inter_block = Generator.intermediate_block(ndim, false)
            model:add(inter_block)
        end
    end

    -- final block.
    local output_block = nn.Sequential()
    output_block:add(UpSampleNearest(2.0))           -- scale up by factor of 2.0
    -- conv1 (3x3)
    output_block:add(SFullConv(ndim, ndim/2, 3, 3, 1, 1, 1, 1))
    if flag_bn then output_block:add(SBatchNorm(ndim/2)) end
    if flag_lrelu then output_block:add(nn.LeakyReLU(0.2,true)) else output_block:add(nn.ReLU(true)) end
    -- conv2 (3x3)
    output_block:add(SFullConv(ndim/2, ndim/2, 3, 3, 1, 1, 1, 1))
    if flag_bn then output_block:add(SBatchNorm(ndim/2)) end
    if flag_lrelu then output_block:add(nn.LeakyReLU(0.2,true)) else output_block:add(nn.ReLU(true)) end
    -- conv3 (1x1)
    output_block:add(SFullConv(ndim/2, nchannel, 1, 1))
    if flag_tanh then output_block:add(nn.Tanh()) end
    --output_block:add(Linear())            -- Linear activation is needed.
    model:add(output_block)

    return model
end

function Generator.intermediate_block(ndim, halving)
    halving = halving or false
    local inter_block = nn.Sequential()
    
    inter_block:add(UpSampleNearest(2.0))           -- scale up by factor of 2.0
    
    if halving then ngf = ndim/2 else ngf = ndim end
    -- conv1
    inter_block:add(SFullConv(ndim, ngf, 3, 3, 1, 1, 1, 1))
    if flag_bn then inter_block:add(SBatchNorm(ngf)) end
    if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
    -- conv2
    inter_block:add(SFullConv(ngf, ngf, 3, 3, 1, 1, 1, 1))
    if flag_bn then inter_block:add(SBatchNorm(ngf)) end
    if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end

    return inter_block
end

return Generator



