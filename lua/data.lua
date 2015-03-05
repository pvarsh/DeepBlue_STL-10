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
-- parse command line arguments
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option('-runlocal', false, 'indicate true if running on local machine')
   cmd:option('-datapath', '../data/a2/stl10_binary/', 'data path for running locally')
   cmd:option('-unlabeled', false, 'do we load unlabeled for unsupervised training')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-yuv', true, 'convert images from RGB to YUV')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print(opt)

print '==> loading dataset'
if opt.runlocal == true then
   path = opt.datapath
else
   path = '/scratch/courses/DSGA1008/A2/binary/'
end

train_X = 'train_X.bin'
train_labels = 'train_y.bin'
test_X = 'test_X.bin'
test_labels = 'test_y.bin'
unlabeled = 'unlabeled_X.bin'

----------------------------------------------------------------------
print '==> loading dataset'
print("Path: ", path .. train_X)

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

if opt.unlabeled == true then
   unlabeled_fd = torch.DiskFile(path .. unlabeled, 'r', true)
   unlabeled_fd:binary():littleEndianEncoding()
   unlabeled_data = torch.ByteTensor(100000,3,96,96)
   unlabeled_data = unlabeled_data:transpose(4,3)
   unlabeled_fd:readByte(unlabeled_data:storage())
end

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

if opt.unlabeled == true then
   unlabeledData = {
      data = unlabeled_data,
      size = function() return unlabeledData.data:size()[1] end
   }
end

if opt.subset == true then
   trainData.data = trainData.data[{ {1,20},{},{},{} }]
   trainData.labels = trainData.labels[{ {1,20} }]
   testData.data = testData.data[{ {1,20},{},{},{} }]
   testData.labels = testData.labels[{ {1,20} }]
end
-- Size variables are used in train and test functions
trsize = trainData:size()
tesize = testData:size()

----------------------------------------------------------------------
print '==> preprocessing data'

-- Convert to Float Tensor
trainData.data = trainData.data:float()
testData.data = testData.data:float()
if opt.unlabeled == true then
   unlabeledData.data = unlabeledData.data:float()
end

-- Convert from RGB to YUV
if opt.yuv == true then
   for i=1,trainData:size() do
      trainData.data[i] = image.rgb2yuv(trainData.data[i])
   end

   for i=1,testData:size() do
      testData.data[i] = image.rgb2yuv(testData.data[i])
   end

-- TODO: Unlabeled conversion not implemented
--       since only a subset of unlabeled data might be used
--       should only preprocess the subset to save resources

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
print '==> zero padding (2px)'

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

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

--------------------------------------------------------------------
-- print '==> visualizing data'

-- -- Visualization is quite easy, using gfx.image().

-- if opt.visualize then
--    first256Samples_y = trainData.data[{ {1,256},1 }]
--    first256Samples_u = trainData.data[{ {1,256},2 }]
--    first256Samples_v = trainData.data[{ {1,256},3 }]
--    gfx.image(first256Samples_y, {legend='Y'})
--    gfx.image(first256Samples_u, {legend='U'})
--    gfx.image(first256Samples_v, {legend='V'})
-- end
