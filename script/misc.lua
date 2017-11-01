-- tools for convenience.


-- Image utils.
function adjust_dyn_range(data, drange_in, drange_out)
    if not drange_in == drange_out then
        local scale = (drange_out[2]-drange_out[1])/(1.0*(drange_in[2]-drange_in[1]))
        local bias = drange_out[1] - drange_in[1]*scale
        data = data:mul(scale):add(bias)
    return data
end
function create_img_grid()
    print('gg')
end


-- Training utils.
function ramp_up()
    print('gg')
end
function rampdown_linear()
    print('gg')
end
function format_time()
    print('gg')
end

-- Logger.
logger = {}
function logger.init(self, filename, data_field)
    require 'optim'
    self.loggerPath = filename
    self.logger =optim.Logger(filename)
    self.logger:setNames(data_field)          -- (e.g.) {'Training acc', 'Test acc.'}
end
function logger.write(self, data)
    self.logger:add(data)
end
function logger.flush(self)
    os.execute(string.format('cat /dev/null > %s', self.loggerPath))
end


