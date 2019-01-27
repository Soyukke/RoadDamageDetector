import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import config
import data_handler

class Detector(nn.Module):
    """ YOLO like model """
    def __init__(self):
        super(Detector, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.grid = config.n_cell
        self.out_dim = config.n_size
        # CNN out_channels = 512
        self.features = vgg16.features
        self.final_dense = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=self.grid*self.grid*self.out_dim, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.final_dense(x)
        x = x.view(-1, self.grid, self.grid, self.out_dim)
        box_predict = torch.sigmoid(x[:, :, :, 0:config.n_offset_class])
        class_predict = F.softmax(x[:, :, :, config.n_offset_class:], dim=3)
        y = torch.cat((box_predict, class_predict), dim=3)
        return y

def loss_function(predict, traindata, device):
    """ YOLO like loss """
    # predict shape (batch_size, n_cell, n_cell, n_boundarybox * 5 + n_class)
    # traindata shape (batch_size, n_boundarybox * 4, n_class)
    x_predict = predict[:, :, :, 0]
    y_predict = predict[:, :, :, 1]
    w_predict = predict[:, :, :, 2]
    h_predict = predict[:, :, :, 3]
    # (-1, config.n_cell, config.n_cell, 4)
    vboxmatrix_predict = predict[:, :, :, 0:4]
    vvbox_predict = np.array([data_handler.boxmatrix2box_v2(boxmatrix) for boxmatrix in vboxmatrix_predict])
    confidence_predict = predict[:, :, :, 4]
    class_predict = predict[:, :, :, config.n_offset_class:]

    x_traindata = traindata[:, :, :, 0]
    y_traindata = traindata[:, :, :, 1]
    w_traindata = traindata[:, :, :, 2]
    h_traindata = traindata[:, :, :, 3]
    vboxmatrix_traindata = traindata[:, :, :, 0:4]
    vvbox_traindata = np.array([data_handler.boxmatrix2box_v2(boxmatrix) for boxmatrix in vboxmatrix_traindata])
    # intersection of union (-1, )
    vIoU = np.zeros(confidence_predict.shape)
    for index_batch in range(vIoU.shape[0]):
        for index_cell_i in range(vIoU.shape[1]):
            for index_cell_j in range(vIoU.shape[2]):
                vIoU[index_batch, index_cell_i, index_cell_j] = data_handler.intersection_over_union(
                    vvbox_predict[index_batch, index_cell_i, index_cell_j], vvbox_traindata[index_batch, index_cell_i, index_cell_j])
    vIoU = torch.from_numpy(np.array(vIoU)).float().to(device)
    class_traindata = traindata[:, :, :, config.n_offset_class:]
    # if object is not exsit, width must be 0
    mask_binary_obj = ((w_traindata != 0)*(h_traindata != 0)).float().to(device)
    mask_binary_noobj = (1 - mask_binary_obj).float().to(device)
    # (batch, config.n_cell, config.n_cell, config.n_class)
    mask_binary_obj_class = mask_binary_obj.view(mask_binary_obj.size(0), mask_binary_obj.size(1), mask_binary_obj.size(2), 1).repeat(1, 1, 1, config.n_class)
    n_obj = max(torch.sum(mask_binary_obj), 1)
    n_noobj = max(torch.sum(mask_binary_noobj), 1)
    # coefficients
    coord = 5
    noobj = 0.5 # no object
    # loss calculations
    boundingbox_center = coord * (F.mse_loss(x_predict*mask_binary_obj, x_traindata*mask_binary_obj, reduction='sum')\
                                  + F.mse_loss(y_predict*mask_binary_obj, y_traindata*mask_binary_obj, reduction='sum'))
    boundingbox_size = coord * (F.mse_loss(torch.sqrt(w_predict)*mask_binary_obj, torch.sqrt(w_traindata)*mask_binary_obj, reduction='sum')\
                                + F.mse_loss(torch.sqrt(h_predict)*mask_binary_obj, torch.sqrt(h_traindata)*mask_binary_obj, reduction='sum'))
    confidence_positive = F.mse_loss(confidence_predict*mask_binary_obj, vIoU*mask_binary_obj, reduction='sum')
    # if cell has no object, confidence to 0
    confidence_negative = noobj * torch.sum((confidence_predict*mask_binary_noobj)**2)
    classify_loss = F.mse_loss(class_predict*mask_binary_obj_class, class_traindata*mask_binary_obj_class)
    return boundingbox_center/n_obj + boundingbox_size/n_obj + confidence_positive/n_obj + confidence_negative/n_noobj + classify_loss/n_obj
