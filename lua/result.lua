----------------------------------------------------------------------
-- A torch script for testing the model for homework 2
-- For DS-GA-1008, Deep Learning
-- New York University
--
-- Written by team Deep Blue (Priyank Bhatia, Emil Christensen, Peter Varshavsky)
-- based on 5_test.lua written by Clement Farabet
-- 03/08/2015
----------------------------------------------------------------------

-- Import libraries
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'      -- provides a normalization operator
require 'image'   -- for color transforms

-- Use 4 threads
torch.setnumthreads(4)

----------------------------------------------------------------------
-- Download files as necessary --

-- Define URLs
model_url = 'http://cims.nyu.edu/~erc399/a2/'
model_file = 'deep_blue_a2_model.net'
mean_file = "mean.dat"
std_file = "std.dat"

data_url = '/scratch/courses/DSGA1008/A2/binary/'
test_file = 'test_X.bin'
train_file = 'test_y.bin'


-- Download files unless already present
if not paths.filep(model_file) then
   print('==> downloading model.net file')
   os.execute('wget ' .. model_url .. model_file)
end
if not paths.filep(train_file) then
   print('==> downloading train data file')
   os.execute('wget ' .. data_url .. train_file)
end
if not paths.filep(test_file) then
   print('==> downloading test data file')
   os.execute('wget ' .. data_url .. test_file)
end


if not paths.filep(mean_file) then
   print('==> downloading training mean file')
   os.execute('wget ' .. model_url .. mean_file)
end
if not paths.filep(std_file) then
   print('==> downloading training std file')
   os.execute('wget ' .. model_url .. std_file)
end
----------------------------------------------------------------------
-- Preprocessing -- 
----------------------------------------------------------------------
print '==> Loading data from bin'

-- Load data
test_fd = torch.DiskFile(path .. test_X, 'r', true)
test_fd:binary():littleEndianEncoding()
test_label_fd = torch.DiskFile(path .. test_labels, 'r', true)
test_label_fd:binary():littleEndianEncoding()

test_data = torch.ByteTensor(8000,3,96,96)
test_fd:readByte(test_data:storage())
test_data = test_data:transpose(4,3)
test_labels = torch.ByteTensor(8000)
test_label_fd:readByte(test_labels:storage())

testData = {
   data = test_data,
   labels = test_labels,
   size = function() return testData.data:size()[1] end
}

-- Convert to float
testData.data = testData.data:float()

print '==> Converting from RGB to YUV'
-- Convert to YUV color
if opt.yuv == true then
   for i=1,testData:size() do
         testData.data[i] = image.rgb2yuv(testData.data[i])
   end
end

print '==> Normalizing globally'
-- Normalize test data using training data mean and std
for i,channel in ipairs(channels) do
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

print '==> Normalizing locally'
-- Normalize test data locally
neighborhood = image.gaussian1D(5)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
for c in ipairs(channels) do
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

-- Zero pad
print '==> Zero padding'
zeroPadder = nn.SpatialZeroPadding(2,2,2,2)
testData['padded'] = torch.FloatTensor(testData:size(), 3, 100, 100)
for i = 1,testData:size() do
   testData.padded[i] = zeroPadder:forward(testData.data[i])
end
testData.data = testData.padded
testData.padded = nil

----------------------------------------------------------------------
-- Evaluate model and save predictions -- 

-- Load the model
print('==> loading model')
model = torch.load(model_file)
model:evaluate()

-- Open file for saving predictions
file = io.open("predictions.csv","w")
file:write("Id,Prediction")

print('==> testing')
-- test over test data
for t = 1,testData:size() do

   -- display progress
   xlua.progress(t, testData:size())

   -- get new sample
   local input = testData.data[t]
   input = input:double()
   local target = testData.labels[t]

   -- test sample
   local pred = model:forward(input)

   -- pick out model's  choice
   m = pred:max()
   idx = torch.linspace(1,pred:size(1),pred:size(1))
   p = idx[pred:eq(m)]

   -- add to predictions
   file:write("\n",t,",",p[1])
end

-- Write to file and close
print '==> writing results to file'
file:flush()
file:close()
