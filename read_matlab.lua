require 'mattorch';
require 'nn';
require 'image';

path = '/scratch/courses/DSGA1008/A2/matlab/'
file_names = {'test.mat', 'train.mat', 'unlabeled.mat'}

test_set = mattorch.load(path..file_names[1])

w = 96
h = 96

x = test_set['X'][{ {1,w*h}, 1 }]
