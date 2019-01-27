import os
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import config

dirname_train = config.dirname_trainimage

def transform_image(image):
    """
    INPUT : image shape (config.width, config.height, 3) BGR only read by cv2\n
    OUTPUT: image shape (3, config.input_width, config.input_height) RGB, and normalized by ImageNet parameters mean and std
    """
    # data tarnsform
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image/255 - config.mean) / config.std
    image = cv2.resize(image, (config.input_width, config.input_height))
    # channels, config.input_height, config.input_width
    image = image.transpose(2, 0, 1)
    return image

def add_box_to_image(image, boxmatrix):
    """
    INPUT : dataset format image and boxmatrix\n
    OUTPUT: return image which are added damage box
    """
    image2 = deepcopy(image)
    vboundarybox = boxmatrix2box(boxmatrix)
    for xmin, ymin, xmax, ymax in vboundarybox:
        image2 = cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=4)
    return image2


def get_location():
    """
    FUNCTION: get dirnames location
    """
    vdirname_location = [os.path.join(dirname_train, dirname_location) for dirname_location in os.listdir(dirname_train)]
    return vdirname_location

def get_filename_label(dirname_location):
    """
    FUNCTION: get xml filenames in dirname_location
    """
    # get filename_label in dirname_location
    dirname_labels = os.path.join(dirname_location, 'labels')
    vfilename_label = [os.path.join(dirname_labels, filename_label) for filename_label in os.listdir(dirname_labels)]
    return vfilename_label

def get_dataset(dirname_location, is_transform=True):
    """
    FUNCTION: dataset format list\n
    OUTPUT : vimage (-1, 3, config.input_width, config.input_height) if is_transform else (-1, config.width, config.height, 3),\n
    vvbox (-1, config.n_cell, config.n_cell, 4)
    """
    vfilename_xml = get_filename_label(dirname_location)
    vimage = []
    vvbox = []
    for filename_xml in vfilename_xml:
        image, vbox = read_xml(filename_xml)
        if is_transform:
            image = transform_image(image)
        vimage.append(image)
        vvbox.append(vbox)
    return vimage, vvbox

def label():
    vdirname_location = [os.path.join(dirname_train, dirname_location) for dirname_location in os.listdir(dirname_train)]
    # xml tag
    boundary = ['xmin', 'xmax', 'ymin', 'ymax']
    # location loop
    for dirname_location in vdirname_location:
        dirname_labels = os.path.join(dirname_location, 'labels')
        vfilename_label = [os.path.join(dirname_labels, filename_label) for filename_label in os.listdir(dirname_labels)]
        # xml loop
        for filename_label in vfilename_label:
            tree = ET.parse(filename_label)
            filename_image = os.path.join(dirname_location, 'images', tree.find('.//filename').text)
            print(filename_image)
            vdamage = tree.findall('.//object')
            # object loop
            for damage in vdamage:
                vboundary = []
                if damage != []:
                    name = damage.find('.//name')
                    for tag in boundary:
                        vboundary.append(damage.find('.//{}'.format(tag)).text)
                    vboundary = np.array(vboundary, dtype=int)
    load_image(filename_image)

def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    return img

def read_xml(filename):
    """
    FUNCTION : read xml file, read damage region and image data. and return array.\n
    INPUT : filename: xml\n
    OUTPUT: image shape (config.width, config.height), vnormalizebox (-1, 4) [x, y, w, h], vcell_index (-1, 2) [cell_i, cell_j]\n
    dataset {(config.input_width, config.input_height), (config.n_cell, config.n_cell, 4)}
    """
    # xml tag
    boundary = ['xmin', 'ymin', 'xmax', 'ymax']

    dirname_location = os.path.dirname(os.path.dirname(filename))
    tree = ET.parse(filename)
    filename_image = os.path.join(dirname_location, 'images', tree.find('.//filename').text)
    vdamage = tree.findall('.//object')

    vnormalizebox = []
    vcell_index = []
    vname = []
    # object loop
    for damage in vdamage:
        box = []
        if damage != []:
            # class is 0 ~ 7
            name = int(damage.find('.//name').text)-1
            # get xmin, ymin, xmax, ymax
            for tag in boundary:
                # xml's xmin, xmax, ymin, ymax is 1 ~ 600 to 0 ~ 599
                box.append(int(damage.find('.//{}'.format(tag)).text)-1)
        normalizebox, cell_index = box2normalizebox(box)
        vnormalizebox.append(normalizebox)
        vcell_index.append(cell_index)
        vname.append(name)

    image = cv2.imread(filename_image, cv2.IMREAD_COLOR)
    vbox = np.zeros([config.n_cell, config.n_cell, config.n_size])
    for index, normalizebox in enumerate(vnormalizebox):
        cell_i, cell_j = vcell_index[index]
        vbox[cell_i, cell_j, 0:4] = normalizebox
        vbox[cell_i, cell_j, config.n_offset_class+vname[index]] = 1

    return image, vbox

def read_xml_test(filename):
    """
    FUNCTION : read xml file, read damage region and image data. and return array.\n
    INPUT : filename: xml\n
    OUTPUT: image shape (config.width, config.height), vnormalizebox (-1, 4) [x, y, w, h], vcell_index (-1, 2) [cell_i, cell_j]\n
    dataset {(config.input_width, config.input_height), (config.n_cell, config.n_cell, 4)}
    """
    # xml tag
    boundary = ['xmin', 'ymin', 'xmax', 'ymax']

    dirname_location = os.path.dirname(os.path.dirname(filename))
    tree = ET.parse(filename)
    filename_image = os.path.join(dirname_location, 'images', tree.find('.//filename').text)
    vdamage = tree.findall('.//object')
    vbox = []
    vname = []
    # object loop
    for damage in vdamage:
        box = []
        if damage != []:
            name = int(damage.find('.//name').text)
            # get xmin, ymin, xmax, ymax
            for tag in boundary:
                # xml's xmin, xmax, ymin, ymax is 1 ~ 600 to 0 ~ 599
                box.append(int(damage.find('.//{}'.format(tag)).text)-1)
        vbox.append(box)
        vname.append(name)
    return filename_image, vbox, vname


def box2normalizebox(box):
    """
    INPUT : box: [xmin, ymin, xmax, ymax]
    OUTPUT: box: [x, y, w, h]
    """
    xmin, ymin, xmax, ymax = box
    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2
    # cell index
    cell_i = int(np.floor(x_center / ((config.width) / config.n_cell)))
    cell_j = int(np.floor(y_center / ((config.height) / config.n_cell)))
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    # normalize
    x = (x_center * config.n_cell / config.width) - cell_i
    y = (y_center * config.n_cell / config.height) - cell_j
    w = width / config.width
    h = height / config.height
    return [x, y, w, h], [cell_i, cell_j]

def normalizebox2box(normalizebox, cell_index):
    """
    INPUT : normalizebox [x, y, w, h], cell_index [cell_i, cell_j]\n
    OUTPUT: box [xmin, ymin, xmax, ymax]
    """
    x, y, w, h = normalizebox
    cell_i, cell_j = cell_index

    x_center = (x + cell_i) * config.width / config.n_cell
    y_center = (y + cell_j) * config.height / config.n_cell
    # width = 1 ~ config.width, height = 1 ~ config.height
    width = max(int(w * config.width), 1)
    height = max(int(h * config.height), 1)

    # x_center = (xmax + xmin) / 2
    # width = xmax - xmin + 1
    # xmax - xmin = width - 1
    # xmax + xmin = 2 * x_center
    # 2*xmax = width - 1 + 2*x_center
    xmax = int((width - 1 + 2*x_center) / 2)
    xmin = xmax + 1 - width
    ymax = int((height- 1 + 2*y_center) / 2)
    ymin = ymax + 1 - height

    # fix xmax = 0 ~ config.width-1, ymax = 0 ~ config.height-1
    xmax = min(xmax, config.width-1)
    ymax = min(ymax, config.height-1)
    # fix xmin = 0 ~ config.width-1, ymax = 0 ~ config.height-1
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    return [xmin, ymin, xmax, ymax]

def boxmatrix2box(boxmatrix, confidence=None):
    """
    INPUT: dataset format damage boxmatrix (config.n_cell, config.n_cell, 4), [x, y, h, w], confidence: (config.n_cell, config.n_cell)
    OUTPUT: vbox list of [xmin, ymin, xmax, ymax]
    """
    vcell_index = []
    vconfidence = []
    for cell_i in range(config.n_cell):
        for cell_j in range(config.n_cell):
            # w and h is not 0, there are boxes
            if (boxmatrix[cell_i, cell_j, 2] != 0) and (boxmatrix[cell_i, cell_j, 3] != 0):
                vcell_index.append([cell_i, cell_j])
                vconfidence.append(confidence[cell_i, cell_j])

    vbox = [normalizebox2box(boxmatrix[cell_i, cell_j], [cell_i, cell_j]) for cell_i, cell_j in vcell_index]
    if confidence is None:
        return vbox
    return vbox, vconfidence


def boxmatrix2box_v2(boxmatrix):
    """
    INPUT: dataset format damage box\n
    OUTPUT: vboxmatrix like list of [xmin, ymin, xmax, ymax]
    """
    boxmatrix2 = np.zeros(boxmatrix.shape)
    vcell_index = []
    for cell_i in range(config.n_cell):
        for cell_j in range(config.n_cell):
            # w or h is not config0 means box is exist
            if (boxmatrix[cell_i, cell_j, 2] != 0) or (boxmatrix[cell_i, cell_j, 3] != 0):
                vcell_index.append([cell_i, cell_j])
                boxmatrix2[cell_i, cell_j] = normalizebox2box(boxmatrix[cell_i, cell_j], [cell_i, cell_j])
    return boxmatrix2


def intersection_over_union(box1, box2):
    """
    Input : box1: [xmin, ymin, xmax, ymax], box2: [xmin, ymin, xmax, ymax]\n
    Output: IoU
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    width1 = xmax1 - xmin1 + 1
    height1 = ymax1 - ymin1 + 1
    width2 = xmax2 - xmin2 + 1
    height2 = ymax2 - ymin2 + 1
    area1 = width1 * height1
    area2 = width2 * height2

    x_min_max = max(xmin1, xmin2)
    y_min_max = max(ymin1, ymin2)
    x_max_min = min(xmax1, xmax2)
    y_max_min = min(ymax1, ymax2)
    # compute the width and height of the bounding box
    intersection_width = max(0, x_max_min - x_min_max + 1)
    intersection_height = max(0, y_max_min - y_min_max + 1)

    area_intersection = intersection_width * intersection_height

    return area_intersection / (area1 + area2 - area_intersection)
