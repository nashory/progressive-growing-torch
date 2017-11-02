-- Generator network structure.

local nn = require 'nn'
local nninit = require 'nninit'
require 'models.custom_layer'


local Generator = {}

local SBatchNorm = nn.SpatialBatchNormalization
local SConv = nn.SpatialConvolution
local SFullConv = nn.SpatialFullConvolution
local UpSampleNearest = nn.SpatialUpSamplingNearest


-- create generator structure.
function Generator.input_block(g_config)
    local flag_bn = g_config['use_bathnorm']
    local flag_lrelu = g_config['use_leakyrelu']
    local nz = g_config['nz']
    local ngf = g_config['fmap_max']
    local nchannel = g_config['num_channels']
    
    -- set input block.
    local input_block = nn.Sequential()
    input_block:add(SFullConv(nz, ngf, 4, 4)
                    :init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}}))
    if flag_bn then input_block:add(SBatchNorm(ngf)) end
    if flag_lrelu then input_block:add(nn.LeakyReLU(0.2,true)) else input_block:add(nn.ReLU(true)) end
    input_block:add(SFullConv(nz, ngf, 3, 3, 1, 1, 1, 1)
                    :init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}}))
    if flag_bn then input_block:add(SBatchNorm(ngf)) end
    if flag_lrelu then input_block:add(nn.LeakyReLU(0.2,true)) else input_block:add(nn.ReLU(true)) end
    
    local nOut  = ngf
    return input_block, nOut
end

function Generator.output_block(ndim, g_config)
    local flag_bn = g_config['use_bathnorm']
    local flag_lrelu = g_config['use_leakyrelu']
    local flag_pxlnorm = g_config['use_pixelnorm']
    local flag_tanh = g_config['use_tanh']
    local nz = g_config['nz']
    local ngf = g_config['fmap_max']
    local nchannel = g_config['num_channels']
    
    -- set output block
    local output_block = nn.Sequential()
    output_block:add(UpSampleNearest(2.0))           -- scale up by factor of 2.0
    -- conv1 (3x3)
    output_block:add(SFullConv(ndim, 16, 3, 3, 1, 1, 1, 1)
                        :init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}}))
    if flag_bn then output_block:add(SBatchNorm(ndim/2)) end
    if flag_lrelu then output_block:add(nn.LeakyReLU(0.2,true)) else output_block:add(nn.ReLU(true)) end
    -- conv2 (3x3)
    output_block:add(SFullConv(16, 16, 3, 3, 1, 1, 1, 1)
                        :init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}}))
    if flag_bn then output_block:add(SBatchNorm(ndim/2)) end
    if flag_lrelu then output_block:add(nn.LeakyReLU(0.2,true)) else output_block:add(nn.ReLU(true)) end
    -- conv3 (1x1)
    output_block:add(SFullConv(16, nchannel, 1, 1)
                        :init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}}))
    if flag_tanh then output_block:add(nn.Tanh()) end
    --output_block:add(Linear())            -- Linear activation is needed.
    return output_block
end


function Generator.intermediate_block(resl, g_config)
    local flag_bn = g_config['use_bathnorm']
    local flag_lrelu = g_config['use_leakyrelu']
    local flag_pxlnorm = g_config['use_pixelnorm']
    local nz = g_config['nz']
    local ngf = g_config['fmap_max']
    local nchannel = g_config['num_channels']
    
    -- (3-->8 / 4-->16 / 5-->32 / 6-->64 / 7-->128 / 8-->256 / 9-->512)
    assert(resl==3 or resl==4 or resl==5 or resl==6 or resl==7 or resl==8 or resl==9)
    
    local halving = false
    local ndim = ngf
    if resl==3 or resl==4 or resl==5 then
        halving = false
        ndim = ngf
    elseif resl==6 or resl==7 or resl==8 or resl==9 then
        halving = true
        for i=1,(resl-5) do ndim = ndim/2 end
    end

    -- set intermediate block
    local inter_block = nn.Sequential()
    inter_block:add(UpSampleNearest(2.0))           -- scale up by factor of 2.0
    
    if halving then
        inter_block:add(SFullConv(ndim*2, ndim, 3, 3, 1, 1, 1, 1)
                            :init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}}))
        if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
        inter_block:add(SFullConv(ndim, ndim, 3, 3, 1, 1, 1, 1)
                            :init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}}))
        if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
    else 
        inter_block:add(SFullConv(ndim, ndim, 3, 3, 1, 1, 1, 1)
                            :init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}}))
        if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
        inter_block:add(SFullConv(ndim, ndim, 3, 3, 1, 1, 1, 1)
                            :init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}}))
        if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
    end
    
    return inter_block, ndim
end
    

return Generator



