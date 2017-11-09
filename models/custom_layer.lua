-- network utilities.
require 'torch'
require 'nn'

-- Pixel-wise normalization layer.
local PixelWiseNorm, parent = torch.class('nn.PixelWiseNorm', 'nn.Module')
function PixelWiseNorm:__init()
    parent.__init(self)
    self.elipson = 1e-8
end
function PixelWiseNorm:updateOutput(input)
    local ndim = input:size(2)          -- batch x ndim x height x width
    local height = input:size(3)
    local norm = torch.sqrt(input:clone():pow(2):mean(2):permute(2,1,3,4))
    self.output = input:cdiv(torch.repeatTensor(norm, ndim, 1, 1, 1):permute(2,1,3,4):add(self.elipson))
    --local norm = torch.sqrt(input:pow(2):sum(3):sum(4)):squeeze():add(self.elipson)
    --self.output = input:cdiv(torch.repeatTensor(norm, height, height,1,1):permute(3,4,1,2))
    
    --local norm = torch.sqrt(input:pow(2):sum(2):div(ndim):add(self.elipson)):squeeze()
    --self.output = input:cdiv(torch.repeatTensor(norm, ndim,1,1,1):permute(2,1,3,4))
    return self.output
end
function PixelWiseNorm:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
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
function MinibatchStatConcat:updateGradInput(input, gradOutput)
    self.gradInput = input:clone():fill(0)
    local nfeature = input:size(2)
    self.gradInput:copy(gradOutput[{{},{1, nfeature},{},{}}])         -- batch x nfeature x h x w
    return self.gradInput
end



-- Resolution selector for fading in new layers during progressinve growing.
local FadeInLayer, parent = torch.class('nn.FadeInLayer', 'nn.Module')
function FadeInLayer:__init(resl_transition_tick)
    parent.__init(self)
    self.transition_tick = resl_transition_tick
    self.alpha = 0
    self.iter = 0
    self.complete = 0
    self.accum = 0              -- accumulated processed images.
end
-- input[1]: from low resolution / input[2]: from high resolution
function FadeInLayer:updateOutput(input)
    assert(type(input)=='table')

    -- multiply and add.
    self.output = torch.add(input[1]:mul(1.0-self.alpha), input[2]:mul(self.alpha))
    return self.output
end
function FadeInLayer:updateAlpha(delta)
    --self.accum = self.accum + batchSize
    -- linear interpolation
    --self.alpha = (self.accum) / (self.transition_tick*1000.0)
    self.alpha = self.alpha + delta
    self.alpha = math.max(0, math.min(1, self.alpha))
    self.complete = (self.alpha)*100.0
end
function FadeInLayer:updateGradInput(input, gradOutput)
    -- init gradInput tensor.
    self.gradInput = {}
    self.gradInput[1] = input[1]:clone():fill(0)
    self.gradInput[2] = input[2]:clone():fill(0)

    --print('alpha:' .. self.alpha)
    --print(self.alpha)
    --print('1-alpha:' .. 1.0-self.alpha)
    self.gradInput[1]:copy(gradOutput:clone():mul(1.0-self.alpha))
    self.gradInput[2]:copy(gradOutput:clone():mul(self.alpha))
    --print('[1] grad sum:' .. gradOutput:sum())
    --print('[2] alpha + 1-alpha:' .. gradOutput:clone():mul(1.0-self.alpha):sum() + gradOutput:clone():mul(self.alpha):sum())
    --self.gradInput[1]:copy(gradOutput)
    --self.gradInput[2]:copy(gradOutput)

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




