require 'torch'
require 'image'
require 'nn'

trainData = torch.load('../data/trainData.dat')

dataset = {}
function dataset:size() return 5000 end

for i=1,dataset:size() do
    dataset[i] = {trainData.data[i], trainData.labels[i]}
end



-- matio = require 'matio'


-- if opt.type == 'float' then
--    print('==> switching to floats')
--    torch.setdefaulttensortype('torch.FloatTensor')
-- elseif opt.type == 'cuda' then
--    print('==> switching to CUDA')
--    require 'cunn'
--    torch.setdefaulttensortype('torch.FloatTensor')
-- end

-- loaded = matio.load('../data/stl10_matlab/train.mat')

-- trainData = {
--    data = loaded.X:transpose(3,4),
--    labels = loaded.y[1],
--    size = function() return trsize end
-- }
