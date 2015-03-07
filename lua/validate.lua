----------------------------------------------------------------------
-- Team Deep Blue
-- 3/2/2015
-- 
-- validation routine
----------------------------------------------------------------------
require 'torch'
require 'xlua'
require 'optim'

----------------------------------------------------------------------
print '==> defining validation set'

validateData = {
   data = trainData.data[{ {(0.9*trainData:size())+1,trainData:size()},{},{},{} }],
   labels = trainData.labels[{ {(0.9*trainData:size())+1,trainData:size()} }],
   size = function() return 0.1*(trainData:size()) end
}

trainData.data = trainData.data[{ {1,0.9*trainData:size()},{},{},{} }]
trainData.labels = trainData.labels[{ {1,trainData:size()} }]
trainData.size = function() return trainData:size() end

----------------------------------------------------------------------
print '==> defining validate procedure'

-- validate function
function validate()
   
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- validate over validation data
   print('==> validating on validation set:')
   for t = 1,validateData:size() do
      -- disp progress
      xlua.progress(t, validateData:size())

      -- get new sample
      local input = validateData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = validateData.labels[t]

      -- validation sample
      local pred = model:forward(input)
      -- print("\n" .. target .. "\n")
      confusion:add(pred, target)

   end

   -- timing
   time = sys.clock() - time
   time = time / validateData:size()
   print("\n==> time to validate 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   validateLogger:add{['% mean class accuracy (validation set)'] = confusion.totalValid * 100}
   if opt.plot then
      validateLogger:style{['% mean class accuracy (validation set)'] = '-'}
      validateLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
