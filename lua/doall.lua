----------------------------------------------------------------------
-- Team Deep Blue
-- 3/2/2015
-- 
-- Process options and repetitively train/test
----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text('Options:')

-- Options copied from Clement Farabet's tutorial
cmd:option('-visualize', true, 'visualize input data and weights during training')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-learningRate', 0.001, 'learning rate at t=0')
cmd:option('-batchSize', 8, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-type', 'cuda', 'type: double | float | cuda')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')

-- Options defined by team
cmd:option('-runlocal', false, 'indicate true if running on local machine')
cmd:option('-yuv', true, 'convert images from RGB to YUV')
cmd:option('-unlabeled', false, 'do we load unlabeled for unsupervised training')
cmd:option('-datapath', '../data/a2/stl10_binary/', 'data path for running locally')
cmd:option('-model', 'cp', 'choose a model to use: cp | a1')
cmd:option('-subset', false, 'subset 20 training and test values for preprocessing testing')

-- Parse options
opt = cmd:parse(arg or {})

----------------------------------------------------------------------
if opt.type == 'float' then
   print('==> defaulting to floats')
   torch.setdefaulttensortype('torch.FloatTensor')

elseif opt.type == 'cuda' then
   print('==> defaulting to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

dofile 'data.lua'
--dofile 'augment.lua'
if opt.model == 'cp' then
    dofile 'model_cp.lua'
end
if opt.model == 'a1' then
    dofile 'model_a1.lua'
end
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
-- Train and test repeatedly
while true do
   train()
   --test()
end
