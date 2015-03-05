----------------------------------------------------------------------
-- Team Deep Blue
-- 3/2/2015
-- 
-- training routine using Stochastic Gradient Descent
----------------------------------------------------------------------
-- use CUDA
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> setting up classes and confusion matrix'

classes = {'1','2','3','4','5','6','7','8','9','10'}
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retreive model parameters
parameters, gradParameters = model:getParameters()

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = 1e-7
  }
optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')


   for t = 1,trainData:size(),opt.batchSize do

      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs  = trainData.data[{{t, math.min(t+opt.batchSize-1, trainData:size())}}]
      local targets = trainData.labels[{{t, math.min(t+opt.batchSize-1, trainData:size())}}]
      
      gradParameters:zero()
      
      local output = model:forward(inputs)
      local f = criterion:forward(output, targets)

      df_do = criterion:backward(output, targets)
      model:backward(inputs, df_do)

      confusion:add(output, targets)

      parameters:add(-opt.learningRate, gradParameters)

   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
