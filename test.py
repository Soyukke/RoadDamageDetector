from copy import deepcopy
import numpy as np
import cv2
import nms
import damage_detector
import data_handler
import config
import torch

def nms_test():
    filename_image = 'input/test/00000039.jpg'
    # make box randomly
    n_box = 96
    image_size = 600
    boxes = np.random.randint(low=0, high=image_size, size=n_box)
    boxes = np.reshape(boxes, newshape=(int(n_box/2), 2))
    boxes = [np.sort(box) for box in boxes]
    boxes = np.reshape(boxes, newshape=(int(n_box/4), 4))
    boxes = boxes[:, [0, 2, 1, 3]]
    # https://note.nkmk.me/python-opencv-draw-function/
    img = cv2.imread(filename_image, cv2.IMREAD_COLOR)
    img_0 = deepcopy(img)
    # add rectangle
    for box in boxes:
        img = cv2.rectangle(img, tuple(box[[0, 2]]), tuple(box[[1, 3]]), color=(255, 0, 0), thickness=4)
    cv2.imwrite('rect_before.jpg', img)
    nms_boxes = nms.non_max_suppression_slow(boxes=boxes, overlapThresh=0.3)
    # add rectangle
    img = deepcopy(img_0)
    for box in nms_boxes:
        img = cv2.rectangle(img, tuple(box[[0, 2]]), tuple(box[[1, 3]]), color=(255, 0, 0), thickness=4)
    # save image nms rectangle
    cv2.imwrite('rect_after.jpg', img)

def box_test():
    box = [10, 100, 90, 150]
    norm_box, cell_index = data_handler.box2normalizebox(box=box)
    print(f'box to norm box {norm_box}, {cell_index}')
    box_ = data_handler.normalizebox2box(norm_box, cell_index)
    print(f'box: {box}, box converted {box_}')

    box1 = [0, 0, 20, 100]
    box2 = [10, 50, 30, 150]
    IoU = data_handler.intersection_over_union(box1, box2)
    print(f'IoU is {IoU}')

    box1 = [0, 0, 20, 100]
    box2 = [0, 0, 20, 100]
    IoU = data_handler.intersection_over_union(box1, box2)
    print(f'IoU is {IoU}')

    box1 = [0, 0, 20, 100]
    box2 = [30, 0, 40, 100]
    IoU = data_handler.intersection_over_union(box1, box2)
    print(f'IoU is {IoU}')

def dataset_test():
    vlocation = data_handler.get_location()
    print(f'locations : {vlocation}')
    vfilename_xml = data_handler.get_filename_label(vlocation[0])
    # print(f'xml filenames : {vfilename_xml}')
    vimage = []
    vvbox = []
    for filename_xml in vfilename_xml:
        image, vbox = data_handler.read_xml(filename_xml)
        vimage.append(image)
        vvbox.append(vbox)
    print(np.shape(vimage), np.shape(vvbox))

def dataset2_test():
    vlocation = data_handler.get_location()
    location = vlocation[0]
    vimage, vvbox = data_handler.get_dataset(location)

def dataset_box_test():
    vlocation = data_handler.get_location()
    location = vlocation[0]
    vimage, vvbox = data_handler.get_dataset(location, is_transform=False)
    image = vimage[0]
    vbox = vvbox[0]
    image2 = data_handler.add_box_to_image(image, vbox)
    cv2.imwrite('damage.jpg', image2)

def loss_test():
    dummy_predict = torch.rand([10, config.n_cell, config.n_cell, 5])
    dummy_train = torch.rand([10, config.n_cell, config.n_cell, 4])
    loss = damage_detector.loss_function(dummy_predict, dummy_train, torch.device('cpu'))
    print(loss)

def name_check():
    vlocation = data_handler.get_location()
    vname_all = []
    for location in vlocation:
        vfilename = data_handler.get_filename_label(location)
        for filename in vfilename:
            filename_image, vbox, vname = data_handler.read_xml_test(filename)
            vname_all.extend(vname)
    vname_unique = np.unique(vname_all)
    print(vname_unique)


if __name__ == '__main__':
    # nms_test()
    # box_test()
    # dataset_test()
    # dataset2_test()
    # dataset_box_test()
    # loss_test()
    name_check()
