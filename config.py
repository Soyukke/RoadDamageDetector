import numpy as np
# train data image size
height = 600
width = 600
# input image size
input_height = 224
input_width = 224
# ImageNet parameters
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
#image = (image.transpose(1, 2, 0) - mean) / std

# network parameters
n_cell = 7
n_boundarybox = 1
n_class = 8 # [1 2 3 4 5 6 7 8]
n_size = 5 * n_boundarybox + n_class
n_offset_class = 5 * n_boundarybox
threshold_intersection_of_union = 0.3

# training parameters
learning_rate = 1e-4
batch_size = 2 
epochs = 200 
cuda = False 
multi_gpu = False 

# location of train datasets
dirname_trainimage = 'input/train'
# location of test datasets
dirname_testimage = 'input/test'
# location of images which are predicted damage box
dirname_testimage_predict = 'results/test'
# location of output model parameters file
fn_model = 'results/model.pt'

# debug mode means using small dataset 
flag_debug = False 
