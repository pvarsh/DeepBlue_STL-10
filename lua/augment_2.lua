print '==> setting up augmentation'
------ Set up size and count variables
n_training = trainData:size() -- original training samples
n_reflected = trainData:size() -- to be generated
n_rotated = trainData:size()-- to be generated
n_data = n_training + n_rotated + n_reflected

n_channels = trainData.data:size()[2]
w = trainData.data:size()[3] -- assuming square
h = trainData.data:size()[4]

------ Create new data and labels tensors to store original
------ training data and augmented data
data = torch.FloatTensor(n_data, n_channels, w, h)
labels = torch.FloatTensor(n_data)

------ Place original training data in new tensors
data[{ {1,n_training} }] = trainData.data
labels[{ {1,n_training} }] = trainData.labels
labels[{ {n_training+1,n_training + n_rotated} }] = trainData.labels
labels[{ {n_training+n_rotated+1,n_training + n_rotated + n_reflected} }] = trainData.labels

print '==> creating rotations'
------ Reflection
for i = 1,n_training do
	image.rotate(data[n_training+i], trainData.data[i],0.35)

end
print("==> hflipping yo")
for i = 1,n_training do
	image.hflip(data[n_training+n_rotated+i], trainData.data[i]) 

end

------ Point trainData.data and trainData.labels to new tensors
trainData.data = data
trainData.labels = labels

-- Size variables are used in train and test functions
trsize = trainData:size()
tesize = testData:size()

print '==> augmentation complete'