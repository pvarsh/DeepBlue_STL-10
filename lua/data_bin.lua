----------------------------------------------------------------------
-- Team Deep Blue
-- 3/2/2015
-- 
-- Load the data and perform preprocessing
----------------------------------------------------------------------
-- Load requirements
require 'torch'
require 'image'
require 'nn'

----------------------------------------------------------------------
print '>> Loading dataset...'
path = '/scratch/courses/DSGA1008/A2/binary/'

train_X = 'train_X.bin'
train_labels = 'train_y.bin'
test_X = 'test_X.bin'
test_labels = 'test_y.bin'
unlabeled = 'unlabeled_X.bin'

----------------------------------------------------------------------
print(">> Data-path: ", path)

train_fd = torch.DiskFile(path .. train_X, 'r', true)
train_fd:binary():littleEndianEncoding()
train_label_fd = torch.DiskFile(path .. train_labels, 'r', true)
train_label_fd:binary():littleEndianEncoding()

test_fd = torch.DiskFile(path .. test_X, 'r', true)
test_fd:binary():littleEndianEncoding()
test_label_fd = torch.DiskFile(path .. test_labels, 'r', true)
test_label_fd:binary():littleEndianEncoding()

train_data = torch.ByteTensor(5000,3,96,96)
train_fd:readByte(train_data:storage())
train_data = train_data:transpose(4,3)
train_labels = torch.ByteTensor(5000)
train_label_fd:readByte(train_labels:storage())

test_data = torch.ByteTensor(8000,3,96,96)
test_fd:readByte(test_data:storage())
test_data = test_data:transpose(4,3)
test_labels = torch.ByteTensor(8000)
test_label_fd:readByte(test_labels:storage())

-- Put data in Lua tables
trainData = {
   data = train_data,
   labels = train_labels,
   size = function() return trainData.data:size()[1] end
}

testData = {
   data = test_data,
   labels = test_labels,
   size = function() return testData.data:size()[1] end
}

-- Size variables are used in train and test functions
trsize = trainData:size()
tesize = testData:size()

----------------------------------------------------------------------
print '>> Preprocessing data...'

-- Convert to Float Tensor
trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- Also convert labels to floats for CUDA compatibility
trainData.labels = trainData.labels:float()
testData.labels = testData.labels:float()

-- Convert from RGB to YUV

for i=1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end

for i=1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Normalization (assuming YUV image)
channels = {'y', 'u', 'v'}
mean = {}
std = {}
------ Normalize each channel globally.
-- Normalize training data
for i, channel in ipairs(channels) do
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i]  = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end
-- Normalize test data using training data mean and std
for i,channel in ipairs(channels) do
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

------ Normalize each channel locally.
-- Define normalization neighborhood and operation
neighborhood = image.gaussian1D(5)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
-- Normalize training data
for c in ipairs(channels) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
end
-- Normalize test data
for c in ipairs(channels) do
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

----------------------------------------------------------------------
print '>> Zero padding (2px)...'

zeroPadder = nn.SpatialZeroPadding(2,2,2,2)
trainData['padded'] = torch.FloatTensor(trainData:size(), 3, 100, 100)
testData['padded'] = torch.FloatTensor(testData:size(), 3, 100, 100)

for i = 1,trainData:size() do
   trainData.padded[i] = zeroPadder:forward(trainData.data[i])
end
for i = 1,testData:size() do
   testData.padded[i] = zeroPadder:forward(testData.data[i])
end

trainData.data = trainData.padded
testData.data  = testData.padded

trainData.padded = nil
testData.padded = nil

----------------------------------------------------------------------
print(">> Saving train data...")
torch.save('../data/trainData.lua',trainData)
trainData = nil

print(">> Saving test data...")
torch.save('../data/testData.lua',testData)
testData = nil
