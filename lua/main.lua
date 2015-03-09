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
    model = 'a1',
    validate = false,
    validateFrac = 0.1,
}

print(opt)

require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
print '>> Running...'

if not paths.filep('../data/trainData.lua') then
    dofile 'data.lua'
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

if opt.validate == true then
    dofile 'validate.lua'
end

dofile 'train.lua'

----------------------------------------------------------------------
-- Train and test repeatedly
while true do
    train()
    
    if opt.validate == true then
        validate()
    end

    -- save model.net every 10 epochs
    if epoch % 10 == 0 then 
        local filename = 'results/model.net'
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('>> Saving model to '..filename..'at epoch #'..epoch)
        torch.save(filename, model)
        dofile 'result.lua'
    end
end
