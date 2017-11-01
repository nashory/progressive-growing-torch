-- script for dealing with network
-- resl: resolution-level (log2 scale)


require 'nn'
require 'models.custom_layer'
local G = require 'models.gen'
--local D = require 'models.dis'


local network = {}


-- grow network
function network.grow_network(gen, dis, resl, g_config)
    -- flush previous fade-in layer first.
    network.flush_FadeInBlock(gen, dis, resl, g_config)
    -- attach new fade-in layer to the last.
    network.attach_FadeInBlock(gen, dis, resl, g_config)
    return gen, dis
end


function network.attach_FadeInBlock(gen, dis, resl, g_config)
    -- generator.
    -- make deep copy of last block and delete it.
    low_res_block = gen.modules[resl-1]:clone()
    gen:remove()
    -- now, make residual block and add fade-in layer.
    local inter_block, ndim = G.intermediate_block(resl, g_config)
    local output_block = G.output_block(ndim, g_config)
    local fadein = nn.Sequential()
    fadein:add( nn.ConcatTable()
                :add(low_res_block)                                         -- for low resl
                :add(nn.Sequential():add(inter_block):add(output_block))    -- for high resl
                )
    fadein:add(nn.FadeInLayer(400))
    gen:add(fadein)

    return gen, dis
end

function network.flush_FadeInBlock(gen, dis, resl, g_config)
    -- remove from generator first.
    -- replace fade-in block with intermediate block.
    -- need to copy weights befroe the removal.
    if resl>3 then 
        high_resl_block = gen.modules[resl-2].modules[1].modules[2]:clone()
        gen:remove()
        gen:add(high_resl_block.modules[1])
        gen:add(high_resl_block.modules[2])
    end
    return gen, dis
end


-- return initial structure of discriminator.
function network.get_init_gen(g_config)
    local model = nn.Sequential()
    local input_block, nOut = G.input_block(g_config)
    model:add(input_block)
    model:add(G.output_block(nOut, g_config))
    return model
end

-- return initial structure of generator.
function network.get_init_dis(g_config)
    print('discriminator')
end

return network


