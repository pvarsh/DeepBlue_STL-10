----------------------------------------------------------------------
-- Team Deep Blue
-- 3/2/2015
-- 
-- testing routine and predictions saver
----------------------------------------------------------------------
require 'torch'
require 'xlua'
require 'optim'

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()

   epoch = epoch or 1
   
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- print('==> opening results file')
   file = io.open("results/"..epoch.."_classifications.csv","w")
   file:write("Id,Category")

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      -- print("\n" .. target .. "\n")
      confusion:add(pred, target)

      -- pick out model's  choice
      m = pred:max()
      pred_flt = pred:float()

      idx = torch.linspace(1,pred:size(1),pred:size(1))
      p = idx[pred_flt:eq(m)]
      file:write("\n",t," , ",p[1])
   end

   -- print '==> writing to results file'
   file:flush()
   file:close()

   -- print '==> defining test procedure'

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
