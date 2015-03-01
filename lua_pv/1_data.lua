----------------------------------------------------------------------
-- This script demonstrates how to load the (SVHN) House Numbers 
-- training data, and pre-process it to facilitate learning.
--
-- The SVHN is a typicaly example of supervised training dataset. 
-- The problem to solve is a 10-class classification problem, similar
-- to the quite known MNIST challenge.
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-runlocal', false, 'indicate true if running on local machine')
   cmd:option('-datapath', '../data/a2/stl10_binary/', 'data path for running locally')
   cmd:option('-unlabeled', false, 'do we load unlabeled for unsupervised training')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-yuv', false, 'convert images from RGB to YUV')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print(opt)

print '==> loading dataset'

-- Here we download dataset files. 

-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.

-- The SVHN dataset contains 3 files:
--    + train: training data
--    + test:  test data
--    + extra: extra training data

-- By default, we don't use the extra training data, as it is much 
-- more time consuming

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

-- ----------------------------------------------------------------------
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




-- for i in 

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

-- trainData.data = trainData.data:float()
-- testData.data = testData.data:float()

-- -- We now preprocess the data. Preprocessing is crucial
-- -- when applying pretty much any kind of machine learning algorithm.

-- -- For natural images, we use several intuitive tricks:
-- --   + images are mapped into YUV space, to separate luminance information
-- --     from color information
-- --   + the luminance channel (Y) is locally normalized, using a contrastive
-- --     normalization operator: for each neighborhood, defined by a Gaussian
-- --     kernel, the mean is suppressed, and the standard deviation is normalized
-- --     to one.
-- --   + color channels are normalized globally, across the entire dataset;
-- --     as a result, each color component has 0-mean and 1-norm across the dataset.

-- -- Convert all images to YUV
-- print '==> preprocessing data: colorspace RGB -> YUV'
-- for i = 1,trainData:size() do
--    trainData.data[i] = image.rgb2yuv(trainData.data[i])
-- end
-- for i = 1,testData:size() do
--    testData.data[i] = image.rgb2yuv(testData.data[i])
-- end

-- -- Name channels for convenience
-- channels = {'y','u','v'}

-- -- Normalize each channel, and store mean/std
-- -- per channel. These values are important, as they are part of
-- -- the trainable parameters. At test time, test data will be normalized
-- -- using these values.
-- print '==> preprocessing data: normalize each feature (channel) globally'
-- mean = {}
-- std = {}
-- for i,channel in ipairs(channels) do
--    -- normalize each channel globally:
--    mean[i] = trainData.data[{ {},i,{},{} }]:mean()
--    std[i] = trainData.data[{ {},i,{},{} }]:std()
--    trainData.data[{ {},i,{},{} }]:add(-mean[i])
--    trainData.data[{ {},i,{},{} }]:div(std[i])
-- end

-- -- Normalize test data, using the training means/stds
-- for i,channel in ipairs(channels) do
--    -- normalize each channel globally:
--    testData.data[{ {},i,{},{} }]:add(-mean[i])
--    testData.data[{ {},i,{},{} }]:div(std[i])
-- end

-- -- Local normalization
-- print '==> preprocessing data: normalize all three channels locally'

-- -- Define the normalization neighborhood:
-- neighborhood = image.gaussian1D(13)

-- -- Define our local normalization operator (It is an actual nn module, 
-- -- which could be inserted into a trainable model):
-- normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- -- Normalize all channels locally:
-- for c in ipairs(channels) do
--    for i = 1,trainData:size() do
--       trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
--    end
--    for i = 1,testData:size() do
--       testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
--    end
-- end

-- ----------------------------------------------------------------------
-- print '==> verify statistics'

-- -- It's always good practice to verify that data is properly
-- -- normalized.

-- for i,channel in ipairs(channels) do
--    trainMean = trainData.data[{ {},i }]:mean()
--    trainStd = trainData.data[{ {},i }]:std()

--    testMean = testData.data[{ {},i }]:mean()
--    testStd = testData.data[{ {},i }]:std()

--    print('training data, '..channel..'-channel, mean: ' .. trainMean)
--    print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

--    print('test data, '..channel..'-channel, mean: ' .. testMean)
--    print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
-- end

-- ----------------------------------------------------------------------
-- -- print '==> visualizing data'
-- -- 
-- -- -- Visualization is quite easy, using gfx.image().
-- -- 
-- -- if opt.visualize then
-- --    first256Samples_y = trainData.data[{ {1,256},1 }]
-- --    first256Samples_u = trainData.data[{ {1,256},2 }]
-- --    first256Samples_v = trainData.data[{ {1,256},3 }]
-- --    gfx.image(first256Samples_y, {legend='Y'})
-- --    gfx.image(first256Samples_u, {legend='U'})
-- --    gfx.image(first256Samples_v, {legend='V'})
-- -- end
