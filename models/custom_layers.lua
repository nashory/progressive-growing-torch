-- network utilities.
require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'


-- Pixel-wise normalization layer.
local PixelWiseNorm, parent = torch.class('nn.PixelWiseNorm', 'nn.Module')
function PixelWiseNorm:__init()
    parent.__init(self)
    self.elipson = 1e-8
end
function PixelWiseNorm:updateOutput(input)
    local ndim = input:size(2)          -- batch x ndim x height x width
    self.output = input:div(torch.sqrt(input:pow(2):sum():div(ndim):add(self.elipson)))
    return self.output
end


-- Minibatch std concatenation (512 --> 513)
local MinibatchStatConcat, parent = torch.class('nn.MinibatchStatConcat', 'nn.Module')
function MinibatchStatConcat:__init()
    parent.__init(self)
end
function MinibatchStatConcat:updateOutput(input)
    -- input state : batch x nfeature x h x w
    local std = input:clone():std(2)            -- batch x 1 x h x w
    self.output = torch.cat(input, std, 2)             -- batch x (nfeature+1) x h x w
    return self.output
end
function MinibatchStatConcat:updateGradInput(input, gradOuptut)
    self.gradInput = input:clone():fill(0)
    local nfeature = input:size(2)
    self.gradInput:copy(gradOutput{{},{1, nfeature},{},{}})         -- batch x nfeature x h x w
    return self.gradInput
end



-- Resolution selector for fading in new layers during progressinve growing.
local FadeInLayer, parent = torch.class('nn.FadeInLayer', 'nn.Module')
function FadeInLayer:__init(lod_transition_kimg, lod_training_kimg)
    parent.__init(self)
    self.lod_transition_kimg = lod_transition_kimg
    self.lod_training_kimg = lod_training_kimg
    self.cur_lod = 0
    self.alpha = 0
end
-- input[1]: from low resolution / input[2]: from high resolution
function FadeInLayer:updateOutput(input, cur_lod)
    assert(type(input)=='table')
    
    -- manipulate alpha according to cur_lod.               !!! NEED TO BE IMPLEMENTED. THIS IS JUST FOR TEST.
    self.alpha = 0.1      -- test

    -- multiply.
    self.output = torch.add(input[1]:mul(1.0-alpha), input[2]:mul(alpha))
    return self.output
end
function FadeInLayer:updateGradInput(input, gradOutput)
    -- init gradInput tensor.
    self.gradInput = {}
    self.gradInput[1] = input[1]:clone():fill(0)
    self.gradInput[2] = input[2]:clone():fill(0)

    self.gradOutput = gradOutput
    self.gradInput[1]:copy(self.gradOutput:clone():mul(1.0-self.alpha))
    self.gradInput[2]:copy(self.gradOuptut:clone():mul(self.alpha))

    return self.gradInput
end




-- brought from: (https://github.com/ryankiros/layer-norm/blob/master/torch_modules/LayerNormalization.lua)
function nn.LayerNorm(nOutput, bias, eps, affine)
    parent.__init(self)
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




function nn.GeneralizedDropOut()
    print('generalized dropout layer.')
end


