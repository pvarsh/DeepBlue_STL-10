print '==> processing options'

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
-- cmd:text()
-- cmd:text('SVHN Dataset Preprocessing')
-- cmd:text()
-- cmd:text('Options:')

-- Options copied from Clement Farabet's tutorial
cmd:option('-visualize', true, 'visualize input data and weights during training')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
-- Options defined by team
cmd:option('-runlocal', false, 'indicate true if running on local machine')
cmd:option('-yuv', true, 'convert images from RGB to YUV')
cmd:option('-unlabeled', false, 'do we load unlabeled for unsupervised training')
cmd:option('-datapath', '../data/a2/stl10_binary/', 'data path for running locally')
-- Parse options
opt = cmd:parse(arg or {})
----------------------------------------------------------------------


-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

dofile '1_data.lua'
dofile '2_model_cp.lua'
dofile '3_loss.lua'
-- dofile '4_train.lua'
