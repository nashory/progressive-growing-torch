-- Generator network structure.

local nn = require 'nn'
local nninit = require 'nninit'
require 'models.custom_layer'


local Generator = {}

local SBatchNorm = nn.SpatialBatchNormalization
local SConv = nn.SpatialConvolution
local SFullConv = nn.SpatialFullConvolution
local UpSampleNearest = nn.SpatialUpSamplingNearest
local PixelWise = nn.PixelWiseNorm
local WN = nn.WeightNorm
local LRN = nn.SpatialCrossMapLRN


-- create generator structure.
function Generator.input_block(g_config)
    local flag_bn = g_config['use_batchnorm']
    local flag_lrelu = g_config['use_leakyrelu']
    local flag_pixel = g_config['use_pixelwise']
    local flag_wn = g_config['use_weightnorm']
    local nz = g_config['nz']
    local ngf = g_config['fmap_max']
    local nchannel = g_config['num_channels']
    
    -- set input block.
    local input_block = nn.Sequential()
    if flag_wn then input_block:add(WN(SFullConv(nz, ngf, 4, 4):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
    else input_block:add(SFullConv(nz, ngf, 4, 4):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
    --if flag_bn then input_block:add(SBatchNorm(ngf)) end
    --if flag_pixel then input_block:add(LRN(2)) end
    if flag_pixel then input_block:add(PixelWise()) end
    if flag_lrelu then input_block:add(nn.LeakyReLU(0.2,true)) else input_block:add(nn.ReLU(true)) end
    
    if flag_wn then input_block:add(WN(SFullConv(ngf, ngf, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
    else input_block:add(SFullConv(ngf, ngf, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
    --if flag_bn then input_block:add(SBatchNorm(ngf)) end
    --if flag_pixel then input_block:add(LRN(2)) end
    if flag_pixel then input_block:add(PixelWise()) end
    if flag_lrelu then input_block:add(nn.LeakyReLU(0.2,true)) else input_block:add(nn.ReLU(true)) end
    
    local ndim  = ngf
    return input_block, ndim
end


function Generator.intermediate_block(resl, g_config)
    local flag_bn = g_config['use_batchnorm']
    local flag_lrelu = g_config['use_leakyrelu']
    local flag_pixel = g_config['use_pixelwise']
    local flag_wn = g_config['use_weightnorm']
    local nz = g_config['nz']
    local ngf = g_config['fmap_max']
    local nchannel = g_config['num_channels']
    
    -- (3-->8 / 4-->16 / 5-->32 / 6-->64 / 7-->128 / 8-->256 / 9-->512)
    assert(resl==3 or resl==4 or resl==5 or resl==6 or resl==7 or resl==8 or resl==9 or resl==10)
    
    local halving = false
    local ndim = ngf
    if resl==3 or resl==4 or resl==5 then
        halving = false
        ndim = ngf
    elseif resl==6 or resl==7 or resl==8 or resl==9 or resl==10 then
        halving = true
        for i=1,(resl-5) do ndim = ndim/2 end
    end

    -- set intermediate block
    local inter_block = nn.Sequential()
    inter_block:add(UpSampleNearest(2.0))           -- scale up by factor of 2.0
    --inter_block:add(nn.UpSampling(2.0, 'nearest'))

    if halving then
        if flag_wn then inter_block:add(WN(SFullConv(ndim*2, ndim, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
        else inter_block:add(SFullConv(ndim*2, ndim, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
        --if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        --if flag_pixel then inter_block:add(LRN(2)) end
        if flag_pixel then inter_block:add(PixelWise()) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end

        if flag_wn then inter_block:add(WN(SFullConv(ndim, ndim, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
        else inter_block:add(SFullConv(ndim, ndim, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
        --if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        --if flag_pixel then inter_block:add(LRN(2)) end
        if flag_pixel then inter_block:add(PixelWise()) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
    
    else 
        if flag_wn then inter_block:add(WN(SFullConv(ndim, ndim, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
        else inter_block:add(SFullConv(ndim, ndim, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
        --if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        --if flag_pixel then inter_block:add(LRN(2)) end
        if flag_pixel then inter_block:add(PixelWise()) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end

        if flag_wn then inter_block:add(WN(SFullConv(ndim, ndim, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
        else inter_block:add(SFullConv(ndim, ndim, 3, 3, 1, 1, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
        --if flag_bn then inter_block:add(SBatchNorm(ndim)) end
        --if flag_pixel then inter_block:add(LRN(2)) end
        if flag_pixel then inter_block:add(PixelWise()) end
        if flag_lrelu then inter_block:add(nn.LeakyReLU(0.2,true)) else inter_block:add(nn.ReLU(true)) end
    end
    
    return inter_block, ndim
end
   
function Generator.to_rgb_block(ndim, g_config)
    local nc = g_config['num_channels']
    local flag_tanh = g_config['use_tanh']
    local flag_pixel = g_config['use_pixelwise']
    local flag_lrelu = g_config['use_leakyrelu']
    local flag_wn = g_config['use_weightnorm']
    
    -- set output block
    local to_rgb_block = nn.Sequential()
    if flag_wn then to_rgb_block:add(WN(SFullConv(ndim, nc, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})))
    else to_rgb_block:add(SFullConv(ndim, nc, 1, 1):init('weight', nninit.kaiming, {gain = {'lrelu', leakiness = 0.2}})) end
    --if flag_pixel then to_rgb_block:add(LRN(1)) end
    --if flag_pixel then to_rgb_block:add(PixelWise()) end
    --if flag_lrelu then to_rgb_block:add(nn.LeakyReLU(0.2,true)) else to_rgb_block:add(nn.ReLU(true)) end
    if flag_tanh then to_rgb_block:add(nn.Tanh()) end 
    return to_rgb_block
end


return Generator



