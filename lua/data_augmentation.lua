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

-- test_fd = torch.DiskFile(path .. test_X, 'r', true)
-- test_fd:binary():littleEndianEncoding()
-- test_label_fd = torch.DiskFile(path .. test_labels, 'r', true)
-- test_label_fd:binary():littleEndianEncoding()

train_data = torch.ByteTensor(5000,3,96,96)
train_fd:readByte(train_data:storage())
train_data = train_data:transpose(4,3)
train_labels = torch.ByteTensor(5000)
train_label_fd:readByte(train_labels:storage())

-- test_data = torch.ByteTensor(8000,3,96,96)
-- test_fd:readByte(test_data:storage())
-- test_data = test_data:transpose(4,3)
-- test_labels = torch.ByteTensor(8000)
-- test_label_fd:readByte(test_labels:storage())

if opt.unlabeled == true then
   unlabeled_fd = torch.DiskFile(path .. unlabeled, 'r', true)
   unlabeled_fd:binary():littleEndianEncoding()
   unlabeled_data = torch.ByteTensor(100000,3,96,96)
   unlabeled_data = unlabeled_data:transpose(4,3)
   unlabeled_fd:readByte(unlabeled_data:storage())
end

-- The last 500 images reserved for validation
trainData = {
   data = train_data[{ {1,4500},{},{},{} }],
   labels = train_labels[{ {1,4500} }],
   size = function() return trainData.data:size()[1] end
}

validationData = {
  data = train_data[{ {4501,5000},{},{},{} }],
  labels = train_labels[{ {4501,5000} }],
  size = function() return validationData.data:size()[1] end
}

-- testData = {
--    data = test_data,
--    labels = test_labels,
--    size = function() return testData.data:size()[1] end
-- }

if opt.unlabeled == true then
   unlabeledData = {
      data = unlabeled_data,
      size = function() return unlabeledData.data:size()[1] end
   }
end

-- if opt.subset == true then
--    trainData.data = trainData.data[{ {1,20},{},{},{} }]
--    trainData.labels = trainData.labels[{ {1,20} }]
--    -- testData.data = testData.data[{ {1,20},{},{},{} }]
--    -- testData.labels = testData.labels[{ {1,20} }]
-- end
-- Size variables are used in train and test functions
-- trsize = trainData:size()
-- tesize = testData:size()


--------------------------------------------------------------------------------
-- Augmentation doc:
-- Takes training set of size N and generates horizontal flips of each image
-- producing a training set of size 2N. Then creates n_folds clones of the resulting
-- training set. The original and flipped images remain unchanged. The remaining
-- (n_folds-1)*2N images are transformed using random rotations, translations, 
-- and color space adjustments in HSV mode.

print('==> Augmenting data')

N = 4500
N_validation = 500

n_folds = 4 -- (n_folds - 1) copies of original and flipped data will be transformed

-- Computing augmented training data tensor shape
train_size = train_data:size()
train_size[1] = N * 2 * n_folds -- mutliplied by two for rotations

print(train_size)

trainData = {
    data = torch.FloatTensor(train_size),
    labels = torch.ByteTensor(N*2*n_folds),
    size = function() return trainData.data:size()[1] end
}

-- The last 500 images reserved for validation

trainData.data[{ {1,4500},{},{},{} }] = train_data[{ {1,4500},{},{},{} }]
trainData.labels[{ {1,4500} }] = train_labels[{ {1,4500} }]

validationData = {
  data = train_data[{ {4501,5000},{},{},{} }],
  labels = train_labels[{ {4501,5000} }],
  size = function() return validationData.data:size()[1] end
}

print(validationData:size())

-- fill labels
for i=1,n_folds*2 do
    trainData.labels[{ {N*(i-1)+1,N*i} }] = train_labels[{ {1,N} }]
end

print(trainData.labels:size())

-- clone unmodified training data to trainData
trainData.data[{ {1,N} }] = train_data[{ {1,N} }]:clone()

print '==> reflections'
-- hflips
for i=1,N do
  trainData.data[N+i] = trainData.data[i]
    -- image.hflip(trainData.data[N+i], trainData.data[i])
end
print '====> cloning folds'
-- clone original and flipped
for i=1,n_folds-1 do
    trainData.data[{ {N*2*i+1,N*2*i+N*2} }] = trainData.data[{ {1,N*2} }] 
end

aug_start = N*2 + 1
aug_end = trainData:size()

print '==> rotations'
-- rotation parameters
num_trans = (n_folds-1) * N*2 -- number of transforms
theta = torch.rand(num_trans):mul(.8):add(-.4)

-- rotate
for i=N*2+1,trainData.size() do
    trainData.data[i] = image.rotate(trainData.data[i], theta[i-N*2])
end

print '==> zoom and translate'
-- magnification and translation parameters
max_translation = 8
shifts = torch.rand(num_trans,2):mul(16):add(-8):int()

-- translate
for i=N*2+1,trainData.size() do
    trainData.data[i] = image.translate(trainData.data[i], shifts[{ i-N*2,1 }], shifts[{ i-N*2,2 }])
    temp = image.scale(trainData.data[i], 108)
    trainData.data[i] = image.crop(temp,6,6,102,102)
end

print '==> HSV transforms'
-- hsv parameters
Hpow = torch.rand(num_trans):mul(.75):add(.75)
Spow = torch.rand(num_trans):add(.5)
Vpow = torch.rand(num_trans):mul(1.5):add(.5)

-- hsv transforms (hue, saturation, value)
for i=N*2+1,trainData.size() do
    temp = image.rgb2hsv(trainData.data[i])
    temp[{1}]:pow(Hpow[i-N*2])
    temp[{2}]:pow(Spow[i-N*2])
    temp[{3}]:pow(Vpow[i-N*2])
    trainData.data[i] = image.hsv2rgb(temp)
end


----------------------------------------------------------------------
print '==> shuffle training data'
shuffle = torch.randperm(N*2*n_folds)
tempData = torch.FloatTensor(trainData.data:size())
tempLabels = torch.ByteTensor(trainData.labels:size())
for i=1,N*2*n_folds do
   tempData[i] = trainData.data[shuffle[i]]
   tempLabels[i] = trainData.labels[shuffle[i]]
end

for i=1,N*2*n_folds do
   trainData.data[i] = tempData[i]
   trainData.labels[i] = tempLabels[i]
end


----------------------------------------------------------------------
print '==> preprocessing data'

-- Convert to Float Tensor
trainData.data = trainData.data:float()
-- testData.data = testData.data:float()
if opt.unlabeled == true then
   unlabeledData.data = unlabeledData.data:float()
end

-- Convert from RGB to YUV
if opt.yuv == true then
   for i=1,trainData:size() do
      trainData.data[i] = image.rgb2yuv(trainData.data[i])
   end

   -- for i=1,testData:size() do
   --    testData.data[i] = image.rgb2yuv(testData.data[i])
   -- end

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
-- for i,channel in ipairs(channels) do
--    testData.data[{ {},i,{},{} }]:add(-mean[i])
--    testData.data[{ {},i,{},{} }]:div(std[i])
-- end

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
-- for c in ipairs(channels) do
--    for i = 1,testData:size() do
--       testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
--    end
-- end

----------------------------------------------------------------------
print '==> zero padding (2px)'

zeroPadder = nn.SpatialZeroPadding(2,2,2,2)
trainData['padded'] = torch.FloatTensor(trainData:size(), 3, 100, 100)
-- testData['padded'] = torch.FloatTensor(testData:size(), 3, 100, 100)

for i = 1,trainData:size() do
   trainData.padded[i] = zeroPadder:forward(trainData.data[i])
end
-- for i = 1,testData:size() do
--    testData.padded[i] = zeroPadder:forward(testData.data[i])
-- end

trainData.data = trainData.padded
-- testData.data  = testData.padded

trainData.padded = nil
-- testData.padded = nil

trainData.labels = trainData.labels:float()
-- testData.labels = testData.labels:float()

----------------------------------------------------------------------
print '==> saving preprocessed images table'

torch.save("augmented_preprocessed_training_set.dat", trainData)
torch.save("train_mean.dat", mean)
torch.save("train_std.dat", std)

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   -- testMean = testData.data[{ {},i }]:mean()
   -- testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   -- print('test data, '..channel..'-channel, mean: ' .. testMean)
   -- print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
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
