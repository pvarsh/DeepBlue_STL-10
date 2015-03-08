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
