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

nfeats = 3        -- number of channels in input images
nstates = 23      -- number of convolution kernels
filtsize = 7      -- width and height of convolution kernels
stepsize = 2      -- stride size for convolution and pooling
-- padding = 2       -- for padding
noutputs = 10     -- number of output classes
pooling = 2       -- width and height of pooling 
drop_prob = 0.5   -- dropout probability (regularization)

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
	model:add(nn.SpatialConvolutionMM(nfeats, nstates, filtsize, filtsize, stepsize, stepsize, padding))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(pooling, pooling, stepsize, stepsize))
 
    model:add(nn.SpatialConvolutionMM(nstates, nstates, filtsize, filtsize, stepsize, stepsize, padding))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(pooling, pooling, stepsize, stepsize))
    
    model:add(nn.View(23*23*23))
    model:add(nn.Linear(23*23*23, 50))
    model:add(nn.ReLU())
    model:add(nn.Linear(50, noutputs))

else 
	model:add(nn.SpatialConvolution(nfeats, nstates, filtsize, filtsize, stepsize, stepsize, padding))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(pooling, pooling, stepsize, stepsize))

    model:add(nn.SpatialConvolution(nstates, nstates, filtsize, filtsize, stepsize, stepsize, padding))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(pooling, pooling, stepsize, stepsize))
    
    model:add(nn.Reshape(23*23*23))
    model:add(nn.Linear(23*23*23, 50))
    model:add(nn.ReLU())
    model:add(nn.Linear(50, noutputs))

end
