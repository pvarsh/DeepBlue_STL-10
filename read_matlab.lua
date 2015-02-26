require 'mattorch';
require 'nn';
require 'image';

path = '/scratch/courses/DSGA1008/A2/matlab/'
file_names = {'test.mat', 'train.mat', 'unlabeled.mat'}

train_set = mattorch.load(path..file_names[2])
test_set = mattorch.load(path..file_names[1])
unlabeled_set = mattorch.load(path..file_names[3])

w = 96
h = 96

test_N = test_set['X']:size()[1]
train_N = train_set['X']:size()[1]

test_set['X'] = test_set['X']:transpose(1, 2):reshape(test_N, 3, 96, 96):transpose(3,4)
train_set['X'] = train_set['X']:transpose(1,2):reshape(train_N, 3, 96, 96):transpose(3,4)
