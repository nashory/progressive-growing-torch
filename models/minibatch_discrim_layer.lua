require 'cudnn'
require 'cunn'

local L1DistanceBatch, parent = torch.class('nn.L1DistanceBatchMat', 'nn.Module')

function L1DistanceBatch:__init()
   parent.__init(self)
   self.inputF = torch.Tensor()
   self.outputF = torch.Tensor()
   self.gradInputF = torch.Tensor()
   self.gradOutputF = torch.Tensor()
   self.buffer1 = torch.Tensor()
   self.buffer2 = torch.Tensor()
end

function L1DistanceBatch:updateOutput(input)
   if input:type()=='torch.CudaTensor' then
      return self:updateOutputGpu(input)
   else
      return self:updateOutputCpu(input)
   end
end

function L1DistanceBatch:updateOutputCpu(input)
   assert(input:nDimension()==3)
   local bs = input:size(1)
   local b = input:size(2)
   local c = input:size(3)
   self.inputF:resize(input:size()):copy(input)
   
   local input = self.inputF
   
   self.outputF:resize(bs,bs,b):zero()
   
   for i=1,bs-1 do
      for j = i+1, bs do
         local x = input[i] - input[j]
         local h = x:norm(1, 2)
         self.outputF[i][j]:copy(h)
         self.outputF[j][i]:copy(h)
      end
   end   

   for i=1,bs do
--      self.outputF[i][i]:fill(1e6)
   end   
   
   self.output:resize(self.outputF:size()):copy(self.outputF)
   
   return self.output
end

function L1DistanceBatch:updateOutputGpu(input)
   assert(input:nDimension()==3)
   local bs = input:size(1)
   local b = input:size(2)
   local c = input:size(3)
   
   self.output:resize(bs,bs,b):zero()
   
   self.output.nn.L1DistanceBatchMat_updateOutput(nil, input, self.output)
   
   return self.output
end

function L1DistanceBatch:updateGradInput(input, gradOutput)
   if input:type()=='torch.CudaTensor' then
      return self:updateGradInputGpu(input, gradOutput)
   else
      return self:updateGradInputCpu(input, gradOutput)
   end
end

function L1DistanceBatch:updateGradInputCpu(input, gradOutput)
   assert(input:nDimension()==3)
   local bs = input:size(1)
   local b = input:size(2)
   local c = input:size(3)
   
   self.inputF:resize(input:size()):copy(input)
   self.gradOutputF:resize(gradOutput:size()):copy(gradOutput)
   
   local input = self.inputF
   local gradOutput = self.gradOutputF
   
   self.gradInputF:resizeAs(self.inputF):zero()
   
   local gradInput = self.gradInputF
   
   local x = self.buffer1:resizeAs(input[1])
   local y = self.buffer2:resizeAs(input[1][1])
   
   for i=1,bs do
      for j = 1, bs do
         local g = gradOutput[i][j]
         torch.sign(x, x:add(input[i], -1, input[j]))
         if i~=j then
            g=g:view(g:size(1), 1):expandAs(x)
            x:cmul(g)
            gradInput[i]:add(x)
            gradInput[j]:add(-1,x)
         end
      end
   end   
   
   self.gradInput:resize(self.gradInputF:size()):copy(self.gradInputF)
   
   return self.gradInput
end

function L1DistanceBatch:updateGradInputGpu(input, gradOutput)
   assert(input:nDimension()==3)
   local bs = input:size(1)
   local b = input:size(2)
   local c = input:size(3)
   
   self.gradInput:resizeAs(input):zero()
   
   self.output.nn.L1DistanceBatchMat_updateGradInput(nil, input, gradOutput, self.gradInput)
   
   return self.gradInput
end

function L1DistanceBatch:type(...)
   parent.type(self, ...)
   return self
end

-- correctness test
if false then
   bs = 5
   d = 4
   e = 3
   input=torch.Tensor(bs,d, e):uniform()
   
   m = nn.L1DistanceBatchMat()
   for i=2, bs do
      input[i]:copy(input[1])
   end
   m:forward(input)
   print(m.output)
   
   g = m.output:clone():uniform()
   m:backward(input,g)
   
   jac = nn.Jacobian
   
   err=jac.testJacobian(m, input)
   print(err)
end

-- speed test
if false then
   require 'cutorch'
   m = nn.L1DistanceBatchMat():cuda()
      bs = 64
   d = 50
   e = 5
   input=torch.Tensor(bs,d,e):uniform():cuda()
      m:forward(input)
   g=m.output:clone():uniform()
   
   for i=1,10 do   
      a = torch.tic()
      m:forward(input)
      print('fwd :', torch.toc(a))
   end
   
   for i=1,100 do   
      a = torch.tic()
      m:backward(input, g)
      print('bwd :', torch.toc(a))
   end
end
