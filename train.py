import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import data_handler
import damage_detector
import config
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if config.cuda else "cpu")
detector = damage_detector.Detector()
if config.multi_gpu:
    n_gpu = torch.cuda.device_count()
    print('Multi GPU mode Use {} GPU'.format(n_gpu))
    detector = nn.DataParallel(detector)
detector.to(device)
optimizer = optim.Adam(params=detector.parameters(), lr=config.learning_rate)

def get_dataset():
    vlocation = data_handler.get_location()
    vimage_all = []
    vvbox_all = []
    for location in vlocation:
        vimage, vvbox = data_handler.get_dataset(location)
        vimage_all.extend(vimage)
        vvbox_all.extend(vvbox)
    return vimage_all, vvbox_all

def get_dataset_debug():
    vlocation = data_handler.get_location()
    location = vlocation[0]
    vimage, vvbox = data_handler.get_dataset(location)
    return vimage[0:20], vvbox[0:20]

if config.flag_debug:
    vimage, vvbox = get_dataset_debug()
else:
    vimage, vvbox = get_dataset()
vimage = torch.from_numpy(np.array(vimage)).float()
vvbox = torch.from_numpy(np.array(vvbox)).float()
datasets = TensorDataset(vimage, vvbox)
trainloader = DataLoader(datasets, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)

train_loss_min = 0
for epoch in range(config.epochs):
    detector.train()
    train_loss = 0
    for batch_idx, (image, vbox) in enumerate(trainloader):
        if config.cuda:
            image = image.to(device)
            vbox = vbox.to(device)
        optimizer.zero_grad()
        predict = detector(image)
        loss = damage_detector.loss_function(predict, vbox, device)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Batch: {batch_idx}, loss: {loss.item()}')

    if (train_loss < train_loss_min) or epoch == 0:
        train_loss_min = train_loss

    if config.multi_gpu:
        torch.save(detector.module.state_dict(), config.fn_model)
    else:
        torch.save(detector.state_dict(), config.fn_model)

    print(f'EpochTotal: {epoch}, loss: {train_loss}')