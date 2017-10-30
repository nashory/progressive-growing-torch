-- network utilities.
require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'

-- Pixel-wise normalization layer.
local PixelWiseNorm, parent = torch.class('nn.PixelWiseNorm', 'nn.Module')
function PixelWiseNorm:__init()
    self.elipson = 1e-8
end
function PixelWiseNorm:updateOutput(input)
    local ndim = input:size(2)          -- batch x ndim x height x width
    return input:div(torch.sqrt(input:pow(2):sum():div(ndim):add(self.elipson)))
end


-- Resolution selector for fading in new layers during progressinve growing.
local LODSelectLayer, parent = torch.class('nn.LODSelectLayer', 'nn.Module')
function LODSelectLayer:__init()
    print('dd')
end
function LODSelectLayer:updateOutput(input)
    print('dd')
end




function nn.WScale()
    print('equalized learning rate for preceding layer.')
end


function nn.MinibatchStatConcat()
    print('minibatch stat concatenation.')
end

function nn.GeneralizedDropOut()
    print('generalized dropout layer.')
end


-- brought from: (https://github.com/ryankiros/layer-norm/blob/master/torch_modules/LayerNormalization.lua)
function nn.LayerNorm(nOutput, bias, eps, affine)
    print('Layer normalization')
    local eps = eps or 1e-5
    local affine = affine or true
    local bias = bias or nil 

    local input = nn.Identity()()
    local mean = nn.Mean(2)(input)
    local mean_rep = nn.Replicate(nOutput,2)(mean) 

    local input_center = nn.CSubTable()({input, mean_rep})
    local std = nn.Sqrt()(nn.Mean(2)(nn.Square()(input_center)))
    local std_rep = nn.AddConstant(eps)(nn.Replicate(nOutput,2)(std))
    local output = nn.CDivTable()({input_center, std_rep})

    if affine == true then
       local biasTransform = nn.Add(nOutput, false)
       if bias ~=nil then
          biasTransform.bias:fill(bias)
       end
       local gainTransform = nn.CMul(nOutput)
       gainTransform.weight:fill(1.)
       output = biasTransform(gainTransform(output))
    end
    return nn.gModule({input},{output})
end
