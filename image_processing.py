import math
import os.path as op
import time

import numpy as np
from pycocotools.coco import COCO
from scipy import misc

data_dir = "dataset"
data_type = "train2014"
ann_file = '{0}/annotations/instances_{1}.json'.format(data_dir, data_type)
ann_file = op.join(
    data_dir, "annotations", "instances_{0}.json".format(data_type))
output_shape = (244, 244, 3)
categories = ["tennis", "boat", "giraffe"]


def process_image(img):
    '''From an image given by the MS Coco API, returns the ndarray standardized
    if the image is tractable, empty list o.w.
        Args:
            - img (dict): the image dictionary returned by the MS Coco API
        Output:
            - ndarray: the matrix associated with the cropped and resized image
    '''
    img_mat = misc.imread(
        op.join(data_dir, data_type, img["file_name"]))
    # We need to crop image on both sides in order to have
    # standardized images in terms of shape
    dimensions = img_mat.shape
    if len(dimensions) == 3:
        bigger_dim = np.argmax(dimensions[:2])
        smaller_dim = 1 - bigger_dim
        difference = dimensions[bigger_dim] -\
            dimensions[smaller_dim]
        if difference > 0:
            left_crop = math.floor(difference / 2)
            right_crop = math.ceil(difference / 2)
            if bigger_dim == 0:
                X = img_mat[left_crop:-right_crop, :, :]
            else:
                X = img_mat[:, left_crop:-right_crop, :]
        else:
            X = img_mat
        return misc.imresize(X, output_shape)
    else:
        return []


def load_images(categories=None):
    '''Preprocessing of the MS Coco dataset
        Output:
            - ndarray (n, 244, 244, 3): the input data (x_train), images
            - list (n): the labels of the images (y_train)
    '''
    X_train = list()
    Y_train = list()
    # initialize COCO api for instance annotations
    coco = COCO(ann_file)
    if categories:
        # load all categories
        cats = coco.loadCats(coco.getCatIds())
        names = [cat["name"] for cat in cats]
    else:
        names = categories
    cat_ids = coco.getCatIds(catNms=names)
    registered_img_ids = set()
    for cat_id in cat_ids:
        # Get all the image ids related to that category
        img_ids = set(coco.catToImgs[cat_id])
        for img_id in img_ids:
            if not(img_id in registered_img_ids):
                # for each image we haven't processed yet, we process it
                img = coco.loadImgs(img_id)[0]
                X = process_image(img)
                if len(X) > 0:
                    X_train = X_train, X
                    Y_train = Y_train, coco.loadCats(cat_id)[0]["name"]
        registered_img_ids = img_ids.union(img_ids)
    X_train = np.array(X_train)
    return X_train, Y_train

load_images(categories)
