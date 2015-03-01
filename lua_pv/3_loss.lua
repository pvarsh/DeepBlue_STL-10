----------------------------------------------------------------------
-- NLL loss with logsoftmax
----------------------------------------------------------------------

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()