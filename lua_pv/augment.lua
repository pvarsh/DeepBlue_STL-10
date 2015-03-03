----------------------------------------------------------------------
-- Generate new training examples:
-- - horizontal reflection (left-right)
-- - rotation TODO
-- - translation TODO
----------------------------------------------------------------------
print '==> setting up augmentation'
------ Set up size and count variables
n_training = trainData:size() -- original training samples
n_reflected = trainData:size() -- to be generated
n_data = n_training + n_reflected

n_channels = trainData.data:size()[2]
w = trainData.data:size()[3] -- assuming square
h = trainData.data:size()[4]

------ Create new data and labels tensors to store original
------ training data and augmented data
data = torch.FloatTensor(n_data, n_channels, w, h)
labels = torch.FloatTensor(n_data)

------ Place original training data in new tensores
data[{ {1,n_training} }] = trainData.data
labels[{ {1,n_training} }] = trainData.labels
labels[{ {n_training+1,n_training+n_reflected} }] = trainData.labels

print '==> creating reflections'
------ Reflection
for i = 1,n_training do
  image.hflip(data[{ {n_training+1, n_training+n_reflected} }], trainData.data)
end

------ Point trainData.data and trainData.labels to new tensors
trainData.data = data
trainData.labels = labels

-- Size variables are used in train and test functions
trsize = trainData:size()
tesize = testData:size()

print '==> augmentation complete'
