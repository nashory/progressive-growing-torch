-- script for dealing with network
-- resl: resolution-level (log2 scale)


require 'nn'
require 'models.custom_layer'
local G = require 'models.gen'
--local D = require 'models.dis'


local network = {}


-- grow network
function network.grow_network(gen, dis, resl, g_config)
    local ngf = g_config.fmap_max           -- nfeatures = 512

    -- flush previous fade-in layer first.
    --network.flush_FadeInBlock(gen)
    --network.flush_FadeInLayer(dis)
    
    -- attach fade-in layer
    --gen:add(G.fadein_block(resl, g_config))
    
    network.attach_FadeInBlock(gen, dis, resl, g_config)

    -- grow generator first.
    --inter_block, ndim = G.intermediate_block(resl, g_config)
    --gen:remove()                            -- remove last layer first,
    --gen:add(inter_block)     -- add intermediate block second,
    --gen:add(G.output_block(ndim, g_config))               -- add output block last.

    -- grow discriminator next.
    -- will be implemented soon.
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

    -- attach fade in layer.
    --gen:add(G.fadein_block(resl, g_config))
    return gen, dis
end

function network.flush_FadeInBlock(gen, dis)
    -- replace fade-in block with intermediate block.
    -- need to copy weights befroe the removal.
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


