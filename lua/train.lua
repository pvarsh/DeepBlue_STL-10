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

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
    --shuffle = torch.randperm(trainData:size())
    --for i = 1, trainData:size() do
    --    trainData.data[i] = trainData.data[{ {shuffle[i]},{},{},{} }]
    --    trainData.labels[i] = trainData.labels[{ {shuffle[i]} }]
    --end

   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      inputs = trainData.data[{ {t,math.min(t+opt.batchSize-1,trainData:size())},{},{},{} }]:cuda()
      targets = trainData.labels[{ {t,math.min(t+opt.batchSize-1,trainData:size())} }]:cuda()

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- estimate f
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- update confusion
         for i = 1,targets:size()[1] do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end
      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
   end

   -- print confusion matrix
   print(confusion)

   -- save/log current net
   --local filename = paths.concat(opt.save, 'model.net')
   --os.execute('mkdir -p ' .. sys.dirname(filename))
   --print('==> saving model to '..filename)
   --torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
