require 'torch'
require 'nn' 

-- sizes
nfeats = 3
width = 32
height = 32

ninputs = nfeats*width*height
nhiddens = ninputs / 2
noutputs = 10

-- model
model = nn.Sequential()

model:add(nn.Linear(ninputs,nhiddens))
model:add(nn.ReLU())
model:add(nn.Linear(nhiddens,noutputs))
model:add(nn.LogSoftMax())

-- train
criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)