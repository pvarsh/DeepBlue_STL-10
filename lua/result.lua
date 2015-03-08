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

-- Use 4 threads
torch.setnumthreads(4)

----------------------------------------------------------------------
-- Download files as necessary --

-- Define URLs
-- model_url = 'http://cims.nyu.edu/~erc399/a2/'
-- model_file = 'deep_blue_a2_model.net'
-- mean_file = "mean.dat"
-- std_file = "std.dat"

-- data_url = '/scratch/courses/DSGA1008/A2/binary/'
-- test_file = 'test_X.bin'
-- train_file = 'test_y.bin'


-- -- Download files unless already present
-- if not paths.filep(model_file) then
--    print('==> downloading model.net file')
--    os.execute('wget ' .. model_url .. model_file)
-- end
-- if not paths.filep(train_file) then
--    print('==> downloading train data file')
--    os.execute('wget ' .. data_url .. train_file)
-- end
-- if not paths.filep(test_file) then
--    print('==> downloading test data file')
--    os.execute('wget ' .. data_url .. test_file)
-- end

----------------------------------------------------------------------
print '>> Loading test data...'

-- Load the data
if not paths.filep('../data/testData.lua') then
    dofile 'data.lua'
    testData = torch.load('../data/testData.lua')
else 
    testData = torch.load('../data/testData.lua')
end

-- Load the model
print('>> Loading model...')
model = torch.load('results/model.net')

----------------------------------------------------------------------
-- Evaluate model and save predictions -- 

model:evaluate()
model:cuda()

classes = {'1','2','3','4','5','6','7','8','9','10'}
confusion = optim.ConfusionMatrix(classes)

-- Open file for saving predictions
file = io.open("results/predictions.csv","w")
file:write("Id,Prediction")

print('>> Testing...')
-- test over test data
for t = 1,testData:size() do

    -- display progress
    xlua.progress(t, testData:size())

    -- get new sample
    local input = testData.data[t]
    input = input:cuda()
    local target = testData.labels[t]

    -- test sample
    local pred = model:forward(input)
    confusion:add(pred, target)

    -- pick out model's  choice
    m = pred:max()
    idx = torch.linspace(1,pred:size(1),pred:size(1))
    p = idx[pred:eq(m)]

    -- add to predictions
    file:write("\n",t,",",p[1])
end

-- print confusion matrix
print(confusion)

-- reset confusion matrix
confusion:zero()

-- Write to file and close
print '>> Writing to results/predictions.csv...'
file:flush()
file:close()