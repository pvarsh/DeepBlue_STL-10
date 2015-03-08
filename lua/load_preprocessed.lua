----------------------------------------------------------------------
-- Team Deep Blue
-- 3/2/2015
-- 
-- Load preprocessed data
----------------------------------------------------------------------
require 'torch'
require 'image'
require 'nn'

trainData = torch.load('augmented_preprocessed_training_set')