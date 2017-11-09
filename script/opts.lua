-- Training parameter settings.
require 'nn'
require 'torch'
require 'optim'

local M = { }

function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Training code.')
	cmd:text()
	cmd:text('Input arguments')


	-------------- Frequently Changed options -----------
	cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
	cmd:option('-name', 'CelebA-HQ_1024', 'experiment name.')
	cmd:option('-snapshot_every', 1000, 'will save model every N tick.')
	
    ---------------- General options ---------------
	cmd:option('-seed', 0, 'random number generator seed to use (0 means random seed)')
	cmd:option('-backend', 'cudnn', 'cudnn option.')


	---------------- Data loading options ---------------
	cmd:option('-data_root_train', '/home1/work/nashory/data/CelebA/Img')
	cmd:option('-nthreads', 8, '# of workers to use for data loading.')
	cmd:option('-display', true, 'true : display server on / false : display server off')
	cmd:option('-display_id', 10, 'display window id.')
	cmd:option('-display_iter', 5, '# of iterations after which display is updated.')
	cmd:option('-display_server_ip', '10.64.81.227', 'host server ip address.')
	cmd:option('-display_server_port', 11200, 'host server port.')
	cmd:option('-save_jpg_iter', 6, 'save every X-th displayed image.')
	cmd:option('-sever_name', 'progressive-growing-gan', 'server name.')


	-------------- Training options---------------
	cmd:option('-lr', 0.0002, 'learning rate')         --0.0002
	cmd:option('-noisetype', 'uniform', 'uniform/normal distribution noise.')

	-- ndims of output features
	cmd:option('-ngf', 512, 'output dimension of first conv layer of generator.')
	cmd:option('-ndf', 512, 'output dimension of first conv layer of discriminator.')
	cmd:option('-nc', 3, 'input image channel. (rgb:3, gray:1)')
	cmd:option('-nz', 512, '# of dimension for input noise(z)')

    --------------- Progressive Growing options -------------
	cmd:option('-transition_tick', 300, 'ticks for transition (1 tick = 1K iter)')               -- (10,5)
	cmd:option('-training_tick', 300, 'ticks for training (1 tick = 1K iter)')
	cmd:option('-total_tick', 1000000, 'ticks for entire training (1 tick = 1K iter)')
    cmd:option('-epsilon_drift', 0.001, 'epsilon_drift')


	cmd:text()

	-- return opt.
	local opt = cmd:parse(arg or {})
	return opt
end

return M


