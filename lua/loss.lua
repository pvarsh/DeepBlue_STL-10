----------------------------------------------------------------------
-- Team Deep Blue
-- 3/2/2015
-- 
-- NLL loss with logsoftmax
----------------------------------------------------------------------

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()