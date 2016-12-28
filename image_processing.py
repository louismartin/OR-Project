import math
import time

import numpy as np
from pycocotools.coco import COCO
from scipy import misc

data_dir = 'data'
data_type = 'train2014'
ann_file = '{0}/annotations/instances_{1}.json'.format(data_dir, data_type)
output_shape = (244, 244, 3)


def load_images():
    '''Preprocessing of the MS Coco dataset
        Output:
            - ndarray (n, 244, 244, 3): the input data (x_train), images
            - list (n): the labels of the images (y_train)
    '''
    x_train = list()
    y_train = list()
    # initialize COCO api for instance annotations
    coco = COCO(ann_file)
    # load all categories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    cat_ids = coco.getCatIds(catNms=nms)
    registered_img_ids = set()
    for cat_id in cat_ids:
        # Get all the image ids related to that category
        img_ids = set(coco.catToImgs[cat_id])
        for img_id in img_ids:

            if not(img_id in registered_img_ids):
                # for each image we haven't processed yet, we process it
                img = coco.loadImgs(img_id)[0]
                img_mat = misc.imread('/'.join(
                    [data_dir, data_type, img['file_name']]))
                # We need to crop image on both sides in order to have
                # standardized images in terms of shape
                dimensions = img_mat.shape
                if len(dimensions) == 3:
                    tic = time.time()
                    bigger_dim = np.argmax(dimensions[:2])
                    smaller_dim = 1 - bigger_dim
                    difference = dimensions[bigger_dim] -\
                        dimensions[smaller_dim]
                    if difference > 0:
                        left_crop = math.floor(difference / 2)
                        right_crop = math.ceil(difference / 2)
                        if bigger_dim == 0:
                            x = img_mat[left_crop:-right_crop, :, :]
                        else:
                            x = img_mat[:, left_crop:-right_crop, :]
                    else:
                        x = img_mat
                    x = misc.imresize(x, output_shape)
                    x_train = x_train, x
                    y_train = y_train, coco.loadCats(cat_id)[0]['name']
                    toc = time.time()
                    print(toc - tic)
        registered_img_ids = img_ids.union(img_ids)
    x_train = np.array(x_train)
    return x_train, y_train

load_images()
