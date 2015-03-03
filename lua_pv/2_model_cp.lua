----------
-- Working to recreate the model Christian Puhrsch (hence cp in file name)
-- used as baseline kaggle submission
----------

----------
-- TODO: CUDA
----------

require 'nn';
require 'image';
require 'optim';
require 'xlua';


----------------------------------------------------------------------
print '==> define parameters'

-- nfeats = 3        -- number of channels in input images
-- nstates = 23      -- number of convolution kernels
-- filtsize = 7      -- width and height of convolution kernels
stepsize = 2      -- stride size for convolution and pooling
-- -- padding = 2       -- for padding
-- noutputs = 10     -- number of output classes
-- pooling = 2       -- width and height of pooling 
drop_prob = 0.5   -- dropout probability (regularization)

-- 10-class problem
noutputs = 10

-- input dimensions
nfeats = 3
width = 100
height = 100
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {200,400,800}
filtsize = 5
poolsize = 2

----------------------------------------------------------------------
-- Christian Puhrsch model architecture:
-- Conv layer (23 channels, 7x7 filters, stride 2, padding 2, RELU activation)
-- Max pooling (3x3 patch, stride 2)
-- Dropout
-- Full connected layer (50 units)
-- Softmax layer

model = nn.Sequential()
-- model:add(nn.SpatialZeroPadding(padding, padding, padding, padding))
if opt.type == 'cuda' then
	-- model:add(nn.SpatialConvolutionMM(nfeats, nstates, filtsize, filtsize, stepsize, stepsize, padding))
 --    model:add(nn.ReLU())
 --    model:add(nn.SpatialMaxPooling(pooling, pooling, stepsize, stepsize))
    
 --    model:add(nn.View(23*23*23))
 --    model:add(nn.Dropout(drop_prob))
 --    model:add(nn.Linear(23*23*23, 50))
 --    model:add(nn.ReLU())
 --    model:add(nn.Linear(50, noutputs))

    -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize, stepsize, stepsize))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize, stepsize, stepsize))

    -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize, stepsize, stepsize))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize, stepsize, stepsize))

    -- stage 3 : standard 2-layer neural network
    model:add(nn.View(nstates[2]*filtsize*filtsize))
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
    model:add(nn.ReLU())
    model:add(nn.Linear(nstates[3], noutputs))

else 
	model:add(nn.SpatialConvolution(nfeats, nstates, filtsize, filtsize, stepsize, stepsize, padding))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(pooling, pooling, stepsize, stepsize))
    
    model:add(nn.Dropout(drop_prob))
    model:add(nn.Linear(23*23*23, 50))
    model:add(nn.ReLU())
    model:add(nn.Linear(50, noutputs))

end
