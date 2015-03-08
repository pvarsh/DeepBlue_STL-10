----------------------------------------------------------------------
-- Team Deep Blue
-- 3/2/2015
-- 
-- Implementation of A1 model architecture
----------------------------------------------------------------------
require 'nn'
require 'image'
require 'optim'
require 'xlua'

----------------------------------------------------------------------
print '==> define parameters'

-- 10-class problem
noutputs = 10

-- input dimensions
nfeats = 3
width = 100
height = 100
ninputs = nfeats*width*height

-- number of hidden units:
nhiddens = ninputs / 2

-- hidden units, filter sizes:
nstates = {200,200,400}
filtsize = 5
poolsize = 2
stepsize = 2

----------------------------------------------------------------------
model = nn.Sequential()

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

