----------------------------------------------------------------------
-- Team Deep Blue
-- 3/8/2015
-- 
-- Process options and repetitively train/test
----------------------------------------------------------------------
print '>> Processing options...'

opt = {
    learningRate = 0.1,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0,
    batchSize = 8,
    validate = true,
    model = 'a1',
    validate = true,
    validateFrac = 0.1,
}

print(opt)

require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
print '>> Running...'

if not paths.filep('../data/trainData.lua') then
    dofile 'data_bin.lua'
    trainData = torch.load('../data/trainData.lua')
else 
    trainData = torch.load('../data/trainData.lua')
end

dofile 'augment.lua'

if opt.model == 'cp' then
    dofile 'model_cp.lua'
end
if opt.model == 'a1' then
    dofile 'model_a1.lua'
end

dofile 'validate.lua'
dofile 'train.lua'

----------------------------------------------------------------------
-- Train and test repeatedly
while true do
    train()
    validate()

    -- save/log current net
    --local filename = paths.concat(opt.save, 'model.net')
    --os.execute('mkdir -p ' .. sys.dirname(filename))
    --print('==> saving model to '..filename)
    --torch.save(filename, model)
end
