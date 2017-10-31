-- script for dealing with network
-- resl: resolution-level (log2 scale)


require 'nn'
local G = require 'models.gen'
--local D = require 'models.dis'


local network = {}


-- grow network
function network.grow_network(gen, dis, resl, g_config)
    local ngf = g_config.fmap_max           -- nfeatures = 512
    -- grow generator first.
    inter_block, ndim = G.intermediate_block(resl, g_config)
    gen:remove()                            -- remove last layer first,
    gen:add(inter_block)     -- add intermediate block second,
    gen:add(G.output_block(ndim, g_config))               -- add output block last.

    -- grow discriminator next.
    -- will be implemented soon.
    
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


