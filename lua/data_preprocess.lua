-----------------
-- Fragment from model_visualize.ipynb
-- To see what preprocessing does to images see model_visualize.ipynb
-- TODO: 
--       load data
--       change data_subset to dataTrain.data and dataTest.data
-----------------


require 'nn';
require 'image';

-- Convert from RGB to YUV
for i=1,data_subset:size()[1] do
    data_subset[i] = image.rgb2yuv(data_subset[i])
end
itorch.image(data_subset[{1}])

-- Normalize each channel globally
channels = {'y', 'u', 'v'}
mean = {}
std = {}
for i, channel in ipairs(channels) do
    mean[i] = data_subset[{ {},i,{},{} }]:mean()
    std[i]  = data_subset[{ {},i,{},{} }]:std()
    data_subset[{ {},i,{},{} }]:add(-mean[i])
    data_subset[{ {},i,{},{} }]:div(std[i])
end

-- Normalize each image locally
neighborhood = image.gaussian1D(5)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

normalization:forward(data_subset[{ 1, {1}, {}, {} }])
for c in ipairs(channels) do
    for i = 1,data_subset:size()[1] do
        data_subset[{ i,{c},{},{} }] = normalization:forward(data_subset[{ i,{c},{},{} }])
    end
end