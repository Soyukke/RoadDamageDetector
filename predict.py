import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import config
import cv2
import damage_detector
import data_handler
import nms

def make_xml(vfilename, vname, vvbox):
    # FUNCTION: output results as xml format
    import xml.etree.ElementTree as ET
    root = ET.Element('annotations')
    for index in range(len(vfilename)):
        annotation = ET.SubElement(root, 'annotation')
        filename = ET.SubElement(annotation, 'filename')
        filename.text = os.path.basename(vfilename[index])
        for index_object in range(len(vname[index])):
            for index_class in range(len(vname[index][index_object])):
                object_tag = ET.SubElement(annotation, 'object')
                name = ET.SubElement(object_tag, 'name')
                name.text = str(vname[index][index_object][index_class])
                pose = ET.SubElement(object_tag, 'pose')
                pose.text = "Unspecified"
                truncated = ET.SubElement(object_tag, 'truncated')
                truncated.text = "0"
                difficult = ET.SubElement(object_tag, 'difficult')
                difficult.text = "0"
                bndbox = ET.SubElement(object_tag, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                ymin = ET.SubElement(bndbox, 'ymin')
                xmax = ET.SubElement(bndbox, 'xmax')
                ymax = ET.SubElement(bndbox, 'ymax')
                # xmin, xmax, ymin, ymax range [0, config.width-1] to [1, config]
                xmin.text = str(vvbox[index][index_object][index_class][0]+1)
                ymin.text = str(vvbox[index][index_object][index_class][1]+1)
                xmax.text = str(vvbox[index][index_object][index_class][2]+1)
                ymax.text = str(vvbox[index][index_object][index_class][3]+1)

    from xml.dom import minidom
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open('results/answer.xml', 'w') as f:
        f.write(xmlstr)


color_set = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 255), (0, 255, 255), (255, 0, 255), (128, 128, 128)]
device = torch.device("cuda" if config.cuda else "cpu")
detector = damage_detector.Detector()
detector.load_state_dict(torch.load(config.fn_model))
if config.multi_gpu:
    n_gpu = torch.cuda.device_count()
    print('Multi GPU mode Use {} GPU'.format(n_gpu))
    detector = nn.DataParallel(detector)
detector.to(device)
detector.eval()

vfilename = [os.path.join(config.dirname_testimage, filename) for filename in os.listdir(config.dirname_testimage)]
vimage = [data_handler.load_image(filename) for filename in vfilename]
# transform for input to net
vimage_transformed = np.array([data_handler.transform_image(image) for image in vimage])

vvbox = []
vname = []
with torch.no_grad():
    for index, image_transformed in enumerate(vimage_transformed):
        vvbox.append([])
        vname.append([])
        image_transformed = torch.from_numpy(image_transformed).float()
        image_transformed = image_transformed.view(1, image_transformed.size(0), image_transformed.size(1), image_transformed.size(2))
        boxmatrix_confidence = detector(image_transformed)
        boxmatrix_confidence = boxmatrix_confidence.cpu()
        boxmatrix = boxmatrix_confidence[0, :, :, 0:4]
        print(f'predicted w h : {boxmatrix[:, :, 2]} {boxmatrix[:, :, 3]}')
        # (config.n_cell, config.n_cell)
        confidence = boxmatrix_confidence[0, :, :, 4]
        class_predict = boxmatrix_confidence[0, :, :, config.n_offset_class:]
        # (config.n_cell, config.n_cell)
        class_predict = torch.argmax(class_predict, dim=2)
        mask_matrix = (confidence > 0.5).float()
        # get class predict box
        image = vimage[index]
        for class_number in range(config.n_class):
            class_mask = (class_predict == class_number).float()
            # w to 0 if confidence <= 0.5
            boxmatrix_temp = boxmatrix.clone()
            boxmatrix_temp[:, :, 2] = boxmatrix[:, :, 2] * mask_matrix * class_mask
            boxmatrix_temp = boxmatrix_temp.numpy()
            vbox, confidence_temp = data_handler.boxmatrix2box(boxmatrix_temp, confidence)
            vbox = np.array(vbox)
            vbox = nms.non_max_suppression_slow(vbox, confidence_temp, config.threshold_intersection_of_union)
            vvbox[index].append(vbox)
            vname[index].append([class_number+1 for _ in range(len(vbox))])
            for xmin, ymin, xmax, ymax in vbox:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color_set[class_number], thickness=4)
        filename_result = os.path.join(config.dirname_testimage_predict, os.path.basename(vfilename[index]))
        cv2.imwrite(filename_result, image)
        print(f'saved {filename_result}')
# make results xml file
make_xml(vfilename, vname, vvbox)
