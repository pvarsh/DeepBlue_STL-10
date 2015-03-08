----------------------------------------------------------------------
-- Team Deep Blue
-- 3/6/2015
-- 
-- Comprehensive script for running a2
----------------------------------------------------------------------
print(">> Loading requirements...")

require 'torch'
require 'image'
require 'nn'
matio = require 'matio'

dataPath = '../data/'
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------

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

print '>> Zero padding (2px)'
zeroPadder = nn.SpatialZeroPadding(2,2,2,2)
trainData['padded'] = torch.FloatTensor(trainData:size(), 3, 100, 100)
for i = 1,trainData:size() do
    trainData.padded[i] = zeroPadder:forward(trainData.data[i])
end
trainData.data = trainData.padded
trainData.padded = nil


print(">> Saving data...")
torch.save('../data/trainData.lua',trainData)
trainData = nil

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


print '>> Zero padding (2px)'
testData['padded'] = torch.FloatTensor(testData:size(), 3, 100, 100)
for i = 1,testData:size() do
   testData.padded[i] = zeroPadder:forward(testData.data[i])
end
testData.data  = testData.padded
testData.padded = nil

print(">> Saving data...")
torch.save('../data/testData.lua',testData)
testData = nil

-- trainData = torch.load('../data/trainData.lua')
-- testData = torch.load('../data/testData.lua')
