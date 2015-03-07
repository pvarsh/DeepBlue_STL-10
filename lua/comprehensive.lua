----------------------------------------------------------------------
-- Team Deep Blue
-- 3/6/2015
-- 
-- Comprehensive script for running a2
----------------------------------------------------------------------
print(">> Loading requirements...")

require 'torch'
require 'image'
require 'xlua'
require 'nn'
--require 'cunn'
matio = require 'matio'

dataPath = '../data/'
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
function preprocess()

    ----------------------------------------
    -- Training Data 
    ----------------------------------------
    print(">> Preprocessing training data...")
    print(">> Loading data...")
    local trainRaw = matio.load(dataPath .. 'train.mat')

    print(">> Reshaping data...")
    trainRaw.X = trainRaw.X:reshape(5000,3,96,96)

    print(">> Transposing data...")
    trainRaw.X = trainRaw.X:transpose(3,4)

    trainData = {
        data = trainRaw.X:float(),
        labels = trainRaw.y:float(),
        size = function() return trainRaw.X:size()[1] end
    }

    print(">> Converting from RGB to YUV...")
    for i=1,trainData:size() do
        trainData.data[i] = image.rgb2yuv(trainData.data[i])
    end

    print(">> Normalizing images globally...")
    channels = {'y', 'u', 'v'}
    mean = {}
    std = {}

    for i, channel in ipairs(channels) do
        mean[i] = trainData.data[{ {},i,{},{} }]:mean()
        std[i]  = trainData.data[{ {},i,{},{} }]:std()
        trainData.data[{ {},i,{},{} }]:add(-mean[i])
        trainData.data[{ {},i,{},{} }]:div(std[i])
    end

    print(">> Normalizing images locally...")
    neighborhood = image.gaussian1D(5)
    normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

    for c in ipairs(channels) do
        for i = 1,trainData:size() do
            trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
        end
    end

    print(">> Saving data...")
    torch.save('../data/trainData.lua',trainData)

    ----------------------------------------
    -- Testing Data 
    ----------------------------------------
    print(">> Preprocessing testing data...")
    print(">> Loading data...")
    local testRaw = matio.load(dataPath .. 'test.mat')

    print(">> Reshaping data...")
    testRaw.X = testRaw.X:reshape(8000,3,96,96)

    print(">> Transposing data...")
    testRaw.X = testRaw.X:transpose(3,4)

    testData = {
        data = testRaw.X:float(),
        labels = testRaw.y:float(),
        size = function() return testRaw.X:size()[1] end
    }

    print(">> Converting from RGB to YUV...")
    for i=1,testData:size() do
        testData.data[i] = image.rgb2yuv(testData.data[i])
    end

    for i,channel in ipairs(channels) do
        testData.data[{ {},i,{},{} }]:add(-mean[i])
        testData.data[{ {},i,{},{} }]:div(std[i])
    end

    print(">> Normalizing images locally...")
    for c in ipairs(channels) do
        for i = 1,testData:size() do
            testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
        end
    end

    print(">> Saving data...")
    torch.save('../data/testData.lua',testData)

end


if not paths.filep('../data/trainData.lua') or not paths.filep('../data/testData.lua') then
    preprocess()
else 
    print(">> Data already preprocessed...")
    print(">> Loading data now...")
    trainData = torch.load('../data/trainData.lua')
    testData = torch.load('../data/testData.lua')
end

----------------------------------------------------------------------
print(">> Defining model...")

nfeats = 3        -- number of channels in input images
nstates = 23      -- number of convolution kernels
filtsize = 7      -- width and height of convolution kernels
stepsize = 2      -- stride size for convolution and pooling
noutputs = 10     -- number of output classes
pooling = 2       -- width and height of pooling 
drop_prob = 0.5   -- dropout probability (regularization)

model = nn.Sequential()

model:add(nn.SpatialConvolution(nfeats, nstates, filtsize, filtsize, stepsize, stepsize, padding))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(pooling, pooling, stepsize, stepsize))

model:add(nn.Dropout(drop_prob))
model:add(nn.Linear(23*23*23, 50))
model:add(nn.ReLU())
model:add(nn.Linear(50, noutputs))

----------------------------------------------------------------------
print(">> Defining loss...")

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
print(">> Training...")

learningRate = 0.001
batchSize = 128
weightDecay = 0.999

parameters,gradParameters = model:getParameters()

function train()
    epoch = epoch or 1
    local clr = 0.1

    --xlua.progress(t, trainData:size())
    --shuffle = torch.randperm(trsize)

    for t = 1, trainData:size(), batchSize do

        local inputs  = trainData.data[{ {t, math.min(t+batchSize-1, trainData:size())}, {}, {}, {} }]
        local targets = trainData.labels[{ {t, math.min(t+batchSize-1, trainData:size())} }]
        
        gradParameters:zero()
        
        local output = model:forward(inputs)
        local f = criterion:forward(output, targets)
        
        local trash, argmax = output:max(2)
        no_wrong = no_wrong + torch.ne(argmax, targets):sum()
        
        model:backward(inputs, criterion:backward(output, targets))
        clr = learningRate * (0.5 ^ math.floor(epoch / weightDecay))
        parameters:add(-clr, gradParameters)
    end

    print(no_wrong)
end


----------------------------------------------------------------------



----------------------------------------------------------------------



----------------------------------------------------------------------



----------------------------------------------------------------------
