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
function logger.init(filename)
    pirnt('dffd')
end
function logger.write(data)
    pirnt('dffd')
end
function logger.flush()
    pirnt('dffd')
end


