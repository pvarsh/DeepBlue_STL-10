----------------------------------------------------------------------
-- Team Deep Blue
-- 3/8/2015
-- 
-- Validation routine
----------------------------------------------------------------------
require 'torch'
require 'xlua'
require 'optim'

----------------------------------------------------------------------
print '>> Defining validation procedure...'

function validate()
    print('>> Validating...')

    -- set model to evaluate mode
    model:evaluate()

    -- validate over validation data
    
    for t = 1,validateData:size() do
        -- disp progress
        xlua.progress(t, validateData:size())

        -- get new sample
        local input = validateData.data[t]
        input = input:cuda()
        local target = validateData.labels[t]

        -- validate sample
        local pred = model:forward(input)
        confusion:add(pred, target)

    end

    -- print confusion matrix
    print(confusion)

    -- reset confusion matrix
    confusion:zero()
end
