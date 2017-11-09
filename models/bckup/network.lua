-- script for dealing with network
-- resl: resolution-level (log2 scale)


require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'math'
require 'models.custom_layer'
local G = require 'models.origin.gen'
local D = require 'models.origin.dis'


local network = {}


-- grow network
function network.grow_network(gen, dis, resl, g_config, d_config, use_cuda)

    local use_cuda = use_cuda or true
    assert(type(use_cuda)=='boolean', 'use_cuda flag = true/false')
    
    -- attach new fade-in layer to the last.
    if resl >2 and resl <= 10 then
        network.attach_FadeInBlock(gen, dis, resl, g_config, d_config)
    end
    if use_cuda then gen:cuda(); dis:cuda(); end
    return gen, dis
end


function network.attach_FadeInBlock(gen, dis, resl, g_config, d_config)
    -- generator.
    -- make deep copy of last block and delete it.
    print(string.format('[From:%d, To:%d] Growing networks ... It might take few seconds... [Generator]',
                                                                            math.pow(2,resl-1), math.pow(2,resl)))
    local transition_tick = g_config['transition_tick']                                                                        
    prev_block = gen.modules[#gen.modules]:clone()
    gen:remove()                            -- remove the last layer, and
    network.freeze_layers(gen)              -- freeze the pretrained blocks.
    
    -- now, make residual block and add fade-in layer.
    local inter_block, ndim = G.intermediate_block(resl, g_config)
    local to_rgb_block = G.to_rgb_block(ndim, g_config)
    local fadein = nn.Sequential()
    fadein:add( nn.ConcatTable()
                :add(nn.Sequential():add(nn.SpatialUpSamplingNearest(2.0)):add(prev_block))     -- for low resl
                :add(nn.Sequential():add(inter_block):add(to_rgb_block)))                       -- for high resl
    fadein:add(nn.FadeInLayer(transition_tick))
    gen:add(fadein)
    fadein = nil

    -- discriminator
    -- make deep copy of first block and delete it.
    print(string.format('[From:%d, To:%d] Growing networks ... It might take few seconds... [Discriminator]',
                                                                            math.pow(2,resl-1), math.pow(2,resl)))
    prev_block = dis.modules[1]:clone()
    dis:remove(1)                           -- remove the first layer, and
    network.freeze_layers(dis)              -- freeze the pretrained blocks.
    
    -- now, make residual block and add fade-in layer.
    local inter_block, ndim = D.intermediate_block(resl, d_config)
    local from_rgb_block = D.from_rgb_block(ndim, d_config)
    local fadein = nn.Sequential()
    fadein:add( nn.ConcatTable()
                :add(nn.Sequential():add(nn.SpatialAveragePooling(2,2,2,2)):add(prev_block))
                :add(nn.Sequential():add(from_rgb_block):add(inter_block)))
    fadein:add(nn.FadeInLayer(transition_tick))
    dis:insert(fadein,1)            -- insert module in front
    fadein = nil

    return gen, dis
end

function network.flush_FadeInBlock(gen, dis, resl, targ)
    -- remove from generator and discriminator.
    -- replace fade-in block with intermediate block.
    -- need to copy weights befroe the removal.
    assert(targ=='gen' or targ=='dis', 'targ argument should be: gen / dis')
    print(string.format('[Res: %d] Flushing fade-in network[%s] ... It might take few seconds...', math.pow(2,resl-1), targ))
    if resl>3 and resl<=11 then
        if targ == 'gen' then
            local high_resl_block = gen.modules[#gen.modules].modules[1].modules[2]:clone()
            gen:remove()
            gen:add(high_resl_block.modules[1]:clone())
            gen:add(high_resl_block.modules[2]:clone())
            
        elseif targ == 'dis' then
            local high_resl_block = dis.modules[1].modules[1].modules[2]:clone()
            dis:remove(1)
            dis:insert(high_resl_block.modules[2]:clone(), 1)
            dis:insert(high_resl_block.modules[1]:clone(), 1)
        end
    end
    return gen, dis
end

function network.freeze_layers(model)
    for i = 1, #model.modules do
        model.modules[i].parameters = function() return nil end     -- freezes the layer when using optim 
        model.modules[i].accGradParameters = function() end         -- overwrite this to reduce computations
    end
end

-- return initial structure of generator.
function network.get_init_gen(g_config)
    local model = nn.Sequential()
    local input_block, ndim = G.input_block(g_config)
    model:add(input_block)
    model:add(G.to_rgb_block(ndim, g_config))
    return model
end

-- return initial structure of discriminator.
function network.get_init_dis(d_config)
    local model = nn.Sequential()
    local output_block, ndim = D.output_block(d_config)
    model:add(D.from_rgb_block(ndim, d_config))
    model:add(output_block)
    return model
end

-- apply equalized learning reate (dynamic weight scaling)
function network.wscale(model)
    function wscale(weight)
        local res = weight:div(torch.sqrt(weight:pow(2):sum()))
        return res
    end
    local nodes = nil
    nodes = model:findModules('nn.SpatialFullConvolution')
    for i=1, #nodes do
        print('----')
        print(nodes[i].weight:mean())
        nodes[i].weight = wscale(nodes[i].weight:clone())
        print(nodes[i].weight:mean())
    end
    --for i=1, #nodes do nodes[i].weight:div(torch.sqrt(nodes[i].weight:pow(2):mean())) end
    nodes = model:findModules('nn.SpatialConvolution')
    for i=1, #nodes do nodes[i].weight = wscale(nodes[i].weight:clone()) end
    --for i=1, #nodes do nodes[i].weight:div(torch.sqrt(nodes[i].weight:pow(2):mean())) end
end


return network


