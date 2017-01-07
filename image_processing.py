import math
import os.path as op
import time

import numpy as np
from pycocotools.coco import COCO
from scipy import misc
from tqdm import tqdm_notebook

data_dir = "dataset"
data_type = "train2014"
ann_file = op.join(
    data_dir, "annotations", "instances_{0}.json".format(data_type))
output_shape = (224, 224, 3)
categories = ["snowboard", "boat", "giraffe"]


def process_image(img_path):
    '''From an image path, returns the ndarray standardized
    if the image is tractable, empty list o.w.
        Args:
            - img_path (str): the path to the image to be processed
        Output:
            - ndarray: the matrix associated with the cropped and resized image
    '''
    img_mat = misc.imread(img_path)
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


def load_images(categories=None, coco=None):
    '''Preprocessing of the MS Coco dataset
        Args:
            - categories (list): list of the categories you want to consider
            - coco (pycoco object): pass a coco instance for faster loading
        Output:
            - ndarray (n, 244, 244, 3): the input data (X_train), images
            - list (n): the labels of the images (Y_train)
            - list (n): the ids of the images
    '''
    X_train = list()
    Y_train = list()
    img_train_ids = list()
    if coco is None:
        # initialize COCO api for instance annotations
        coco = COCO(ann_file)
    if categories:
        # here we only take a subset of the categories
        names = categories
    else:
        # load all categories
        cats = coco.loadCats(coco.getCatIds())
        names = [cat["name"] for cat in cats]
    cat_ids = coco.getCatIds(catNms=names)
    registered_img_ids = set()
    for cat_id in tqdm_notebook(cat_ids):
        # Get all the image ids related to that category
        img_ids = set(coco.catToImgs[cat_id])
        for img_id in tqdm_notebook(img_ids, position=1):
            if not(img_id in registered_img_ids):
                # for each image we haven't processed yet, we process it
                img = coco.loadImgs(img_id)[0]
                img_path = op.join(data_dir, data_type, img["file_name"])
                X = process_image(img_path)
                if len(X) > 0:
                    X_train = X_train + [X]
                    Y_train = Y_train + [cat_id]
                    img_train_ids = img_train_ids + [img_id]
        registered_img_ids = img_ids.union(img_ids)
    X_train = np.array(X_train)
    return X_train, Y_train, img_ids
