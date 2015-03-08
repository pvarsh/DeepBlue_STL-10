----------------------------------------------------------------------
-- Team Deep Blue
-- 3/8/2015
-- 
-- Augment data
----------------------------------------------------------------------
require 'image'

if opt.validate == true then
    print '>> Creating validation set...'

    oldSize = trainData:size()

    validateData = {
       data = trainData.data[{ {((1.0-opt.validateFrac)*oldSize)+1,oldSize},{},{},{} }],
       labels = trainData.labels[{ {((1.0-opt.validateFrac)*oldSize)+1,oldSize} }],
       size = function() return opt.validateFrac*oldSize end
    }

    trainData = {
       data = trainData.data[{ {1,(1.0-opt.validateFrac)*oldSize},{},{},{} }],
       labels = trainData.labels[{ {1,(1.0-opt.validateFrac)*oldSize} }],
       size = function() return (1.0-opt.validateFrac)*oldSize end
    }
end

----------------------------------------------------------------------
print '>> Beginning augmentation...'

-- Set up size and count variables
n_training = trainData:size() -- original training samples
n_reflected = trainData:size() -- to be generated
n_rotated = trainData:size()-- to be generated
n_data = n_training + n_rotated + n_reflected

n_channels = trainData.data:size()[2]
w = trainData.data:size()[3]
h = trainData.data:size()[4]

-- Create new data and labels tensors to store original
-- training data and augmented data
data = torch.FloatTensor(n_data, n_channels, w, h)
labels = torch.FloatTensor(n_data)

-- Place original training data in new tensors
data[{ {1,n_training} }] = trainData.data
labels[{ {1,n_training} }] = trainData.labels
labels[{ {n_training+1,n_training + n_rotated} }] = trainData.labels
labels[{ {n_training+n_rotated+1,n_training + n_rotated + n_reflected} }] = trainData.labels

print '>> Creating rotations...'
-- Reflection
for i = 1,n_training do
	image.rotate(data[n_training+i], trainData.data[i],0.35)

end
print(">> Creating horizontal flips...")
for i = 1,n_training do
	image.hflip(data[n_training+n_rotated+i], trainData.data[i]) 

end

-- Point trainData.data and trainData.labels to new tensors
trainData.data = data
trainData.labels = labels
trainData.size = function() return n_data end

print '>> Augmentation complete...'
