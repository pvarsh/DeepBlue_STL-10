----------
-- Working to recreate the model Christian Puhrsch (hence cp in file name)
-- used as baseline kaggle submission
----------

----------
-- Will start working to get this into the pipeline after lunch today (Sunday)
-- so if you want to make changes, please do in separate branch or 
-- copy file to separate folder
---------- Peter

require 'nn';
require 'image';
require 'optim';
require 'xlua';

model = nn.Sequential()

nfeats = 3
nstates = 23
filtsize = 7
stepsize = 2
padding = 2
noutputs = 10

pooling = 2
drop_probability = 0.5

-- nInputPlane, nOutputPlane, kW, kH, dW, dH, padding
model:add(nn.SpatialZeroPadding(padding, padding, padding, padding))
model:add(nn.SpatialConvolution(nfeats, nstates, filtsize, filtsize, stepsize, stepsize, padding))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(pooling, pooling, stepsize, stepsize))
model:add(nn.Dropout(drop_probability))
model:add(nn.Reshape(23*23*23))
model:add(nn.Linear(23*23*23, 50))
model:add(nn.ReLU())
model:add(nn.Linear(50, noutputs))