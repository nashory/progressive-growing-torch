-- Discriminator network structure.

local nn = require 'nn'
local nninit = require 'nninit'
require 'models.custom_layer'


local Discrim = {}

local SBatchNorm = nn.SpatialBatchNormalization
local SConv = nn.SpatialConvolution
local MiniBatch = nn.MinibatchStatConcat
local AvgPool = nn.SpatialAveragePooling
local MaxPool = nn.SpatialMaxPooling
local PixelWise = nn.PixelWiseNorm
local LRN = nn.SpatialCrossMapLRN
local WN = nn.WeightNorm



function Discrim.output_block(d_config)
    local flag_bn = d_config['use_batchnorm']
    local flag_lrelu = d_config['use_leakyrelu']
    local flag_pixel = d_config['use_pixelwise']
    local flag_wn =  d_config['use_weightnorm']
    local ndf = d_config['fmap_max']
    
    -- set output block
    local output_block = nn.Sequential()
    output_block:add(MiniBatch())
    -- conv1 (3x3)
    if flag_wn then output_block:add(WN(SConv(ndf+1, ndf, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
    else output_block:add(SConv(ndf+1, ndf, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
    if flag_bn then output_block:add(SBatchNorm(ndf)) end
    if flag_lrelu then output_block:add(nn.LeakyReLU(0.2,true)) else output_block:add(nn.ReLU(true)) end
    -- conv2 (4x4)
    if flag_wn then output_block:add(WN(SConv(ndf, ndf, 4, 4):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
    else output_block:add(SConv(ndf, ndf, 4, 4):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
    if flag_bn then output_block:add(SBatchNorm(ndf)) end
    if flag_lrelu then output_block:add(nn.LeakyReLU(0.2,true)) else output_block:add(nn.ReLU(true)) end

    -- conv3 (1x1)
    if flag_wn then output_block:add(WN(SConv(ndf, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
    else output_block:add(SConv(ndf, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
    if flag_lrelu then output_block:add(nn.LeakyReLU(0.2,true)) else output_block:add(nn.ReLU(true)) end
    output_block:add(nn.View(1))          -- batchsize x 1
    output_block:add(nn.Sigmoid())
    -- Linear
    --output_block:add(nn.View(ndf))          -- batchsize x ndf
    --output_block:add(nn.Linear(ndf, 1))
    --output_block:add(nn.Sigmoid())
    return output_block, ndf
end


function Discrim.intermediate_block(resl, d_config)
    local flag_bn = d_config['use_batchnorm']
    local flag_lrelu = d_config['use_leakyrelu']
    local flag_pixel = d_config['use_pixelwise']
    local flag_wn =  d_config['use_weightnorm']
    local ndf = d_config['fmap_max']
    
    -- (3-->8 / 4-->16 / 5-->32 / 6-->64 / 7-->128 / 8-->256 / 9-->512)
    assert(resl==3 or resl==4 or resl==5 or resl==6 or resl==7 or resl==8 or resl==9 or resl==10)
    
    local halving = false
    local ndim = ndf
    if resl==3 or resl==4 or resl==5 then
        halving = false
        ndim = ndf
    elseif resl==6 or resl==7 or resl==8 or resl==9 or resl==10 then
        halving = true
        for i=1,(resl-5) do ndim = ndim/2 end
    end

    -- set intermediate block
    local inter_block = nn.Sequential()
    if halving then
        if flag_wn then inter_block:add(WN(SConv(ndim, ndim, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
        else inter_block:add(SConv(ndim, ndim, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
        if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end

        if flag_wn then inter_block:add(WN(SConv(ndim, ndim*2, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
        else inter_block:add(SConv(ndim, ndim*2, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
        if flag_bn then inter_block:add(SBatchNorm(ndim*2)) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
        inter_block:add(AvgPool(2,2,2,2))                   -- downsample by factor of 2.0
    else 
        if flag_wn then inter_block:add(WN(SConv(ndim, ndim, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
        else inter_block:add(SConv(ndim, ndim, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
        if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
        
        if flag_wn then inter_block:add(WN(SConv(ndim, ndim, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
        else inter_block:add(SConv(ndim, ndim, 3, 3, 1, 1, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
        if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
        inter_block:add(AvgPool(2,2,2,2))                   -- downsample by factor of 2.0
    end
    
    return inter_block, ndim
end
    
function Discrim.from_rgb_block(ndim, d_config)
    local nc = d_config['num_channels']
    local flag_lrelu = d_config['use_leakyrelu']
    local flag_pixel = d_config['use_pixelwise']
    local flag_wn =  d_config['use_weightnorm']
    local flag_bn = d_config['use_batchnorm']
    
    -- set input block.
    local from_rgb_block = nn.Sequential()
    if flag_wn then from_rgb_block:add(WN(SConv(nc, ndim, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
    else from_rgb_block:add(SConv(nc, ndim, 1, 1):noBias():init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
    if flag_bn then from_rgb_block:add(SBatchNorm(ndim)) end
    if flag_lrelu then from_rgb_block:add(nn.LeakyReLU(0.2,true)) else from_rgb_block:add(nn.ReLU(true)) end
    return from_rgb_block
end

return Discrim



