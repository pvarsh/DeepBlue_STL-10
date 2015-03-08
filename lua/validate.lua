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

oldSize = trainData:size()

validateData = {
   data = trainData.data[{ {(0.9*oldSize)+1,oldSize},{},{},{} }],
   labels = trainData.labels[{ {(0.9*oldSize)+1,oldSize} }],
   size = function() return 0.1*oldSize end
}

trainData = {
   data = trainData.data[{ {1,0.9*oldSize},{},{},{} }],
   labels = trainData.labels[{ {1,0.9*oldSize} }],
   size = function() return 0.9*oldSize end
}

----------------------------------------------------------------------
print '==> defining validate procedure'

-- validate function
function validate()
   
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
      confusion:add(pred, target)

   end

   -- print confusion matrix
   print(confusion)

   -- next iteration:
   confusion:zero()
end
