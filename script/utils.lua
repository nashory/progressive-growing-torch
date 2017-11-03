-- tools for convenience.
require 'optim'
require 'image'


-- Image utils.
function adjust_dyn_range(data, drange_in, drange_out)
    if not drange_in == drange_out then
        local scale = (drange_out[2]-drange_out[1])/(1.0*(drange_in[2]-drange_in[1]))
        local bias = drange_out[1] - drange_in[1]*scale
        data = data:mul(scale):add(bias)
    end
    return data
end

function create_img_grid(input, unitSize, len)
    -- input: Tensor(batch x channel x height x width)
    -- gridsize: 3 x (len x unitSize) x (len x unitSize)
    local len = len or math.floor(math.sqrt(input:size(1)))
    local batch = math.pow(len, 2)
    local unitSize = unitSize or 64
    
    -- convert from [-1,1] to [0,1] pixel range.
    input:add(1):div(2)

    -- draw grid.
    local grid = torch.Tensor(3, len*unitSize, len*unitSize):zero()
    local cnt = 1
    for h=1, len*unitSize, unitSize do
        for w=1, len*unitSize, unitSize do
            if cnt <= input:size(1) then
                grid[{{},{h, h+unitSize-1},{w,w+unitSize-1}}]:copy(size_resample(input[{{cnt},{},{},{}}]:squeeze(), unitSize))
            else 
                grid[{{},{h, h+unitSize-1},{w,w+unitSize-1}}]:fill(1)
            end
            cnt = cnt + 1
        end
    end
    return grid
end

function size_resample(im, targSize)
    return image.scale(im:float():clone(), targSize, targSize, 'simple')
end


-- Logger.
local logger = {}
function logger.init(self, filename, data_field)
    require 'optim'
    self.loggerPath = filename
    self.logger = optim.Logger(filename)
    self.logger:setNames(data_field)          -- (e.g.) {'Training acc', 'Test acc.'}
end
function logger.write(self, data)
    self.logger:add(data)
end
function logger.flush(self)
    os.execute(string.format('cat /dev/null > %s', self.loggerPath))
end

