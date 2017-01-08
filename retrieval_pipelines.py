from collections import Counter
import os.path as op
import re
import string

import numpy as np
import pandas as pd

from image_processing import process_image, data_dir, data_type
from vgg import compute_nn_features
from word2vec import sentence2vec


def absolute_coco_path(img_id, coco):
    '''From an MS Coco image ID returns the absolute path of the image
        Args:
            - img_id (str): the ID of the image in the MS Coco DB
            - coco (MS Coco API): the API to link the img ids to MS Coco images
        Output:
            - str: the absolute path of the image
    '''
    img = coco.loadImgs(img_id)[0]
    return op.join(data_dir, data_type, img["file_name"])


def tag_to_image_search(tag_features, W_text, database_images, img_ids,
                        coco, n_images=3):
    '''From a given tag returns the top n_images in the database closest to the
    tag in the common feature space.
        Args:
            - tag_features (ndarray): the features of the tag to search
            - W_text (ndarray): the matrix of transition between the textual
            feature space to the common space.
            - database_images (ndarray): an array containing the features of
            the database images in the common space.
            - img_ids (list): the list of the image IDs of the database_images.
            - coco (MS Coco API): the API to link the img ids to MS Coco images
            - n_images (int): the number of images to retrieve
        Output:
            - list: the list of the absolute paths of the retrieved images
    '''
    # Put the tag in the common space
    common_space_features = W_text.dot(tag_features)
    # In the common space find its nearest neighbours
    idx_nearest_neigh = nearest_neighbours(
        common_space_features, database_images, n_images)
    return img_ids[idx_nearest_neigh]


def image_to_tag_search(visual_features, W_image, database_captions,
                        img_ids, coco, n_tags=3, expanding_factor=10):
    '''From a given image (by its vgg19 features), returns the top
    n_tags*expanding_factor annotations in the database closest to the image in
    the common space.
        Args:
            - visual_features (ndarray): the features of the image to search
            - W_image (ndarray): the matrix of transition between the visual
            feature space to the common space.
            - database_captions (ndarray): an array containing the features of
            the database tags in the common space.
            - img_ids (list): the list of the image IDs of the
            database_captions.
            - coco (MS Coco API): the API to link the img ids to MS Coco images
            - n_tags (int): the number of tags to retrieve eventually
            - expanding_factor (int): the proportion of captions we need to
            retrieve to achieve the retrieval of n_tags tags.
        Output:
            - pd data frame: the the data frame regrouping the captions of the
            n_tags*expanding_factor closest images in the database.
    '''
    # Put the tag in the common space
    common_space_features = W_image.dot(visual_features)
    # In the common space find its nearest neighbours (we take a lot of them to
    # then select the most common tags)
    idx_nearest_neigh = nearest_neighbours(
        common_space_features, database_captions, 10*n_tags)
    img_ids = img_ids[idx_nearest_neigh]
    # We load the annotations
    dataframe_path = op.join(data_dir, 'annotations', 'captions.csv')
    captions_df = pd.read_csv(dataframe_path, index_col=0)
    return captions_df.loc[img_ids]


def most_common_tags(annotations, n_tags, stopwords):
    '''From a data frame of annotations, retrieves the n_tags most common tags.
        Args:
            - annotations (pd data frame): the data frame regrouping captions
            - n_tags (int): the number of tags you want to retrieve
            - stopwords (iterable): the list of words we don't want to take
            into account
        Output:
            - list: the list of the n_tags most common tags in the annotations
    '''
    regex = re.compile('[\W_ ]+')
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    counter = Counter()
    for caption in annotations["caption"]:
        sentence = regex.sub('', caption)
        sentence = sentence.lower()
        words = sentence.split(' ')
        counter.update([word for word in words
                        if word not in stopwords and word != ""])
    return counter.most_common(n_tags)


def nearest_neighbours(new_X, X, k):
    '''Returns the indices of the k nearest neighbours of new_X in X.
        Args:
            - new_X (ndarray): the vector whose nearest neighbours you want to
            find.
            - X (ndarray): the vectors in which you are looking for neighbours.
            - k (int): the number of neighbours you want to return.
        Output:
            - list: the indices of the nearest neighbours
    '''
    dist = np.linalg.norm(X - new_X, axis=1)
    return np.argpartition(dist, k)[:k]
