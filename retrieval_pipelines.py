import os.path as op

import numpy as np

from image_processing import process_image
from vgg import compute_nn_features
from word2vec import sentence2vec


def absolute_coco_path(img_id, coco):
    '''From an MS Coco imame ID returns the absolute path of the image
        Args:
            - img_id (str): the ID of the image in the MS Coco DB
            - coco (MS Coco API): the API to link the img ids to MS Coco images
        Output:
            - str: the absolute path of the image
    '''
    img = coco.loadImgs(img_id)[0]
    return op.join(data_dir, data_type, img["file_name"])


def tag_to_image_search(tag, W_text, word_model, database_images, img_ids,
                        coco, n_images=3):
    '''From a given tag returns the top n_images in the database closest to the
    tag in the common feature space.
        Args:
            - tag (str): the tag to search
            - W_text (ndarray): the matrix of transition between the textual
            feature space to the common space.
            - word_model (gensim model): the word 2 vec model
            - database_images (ndarray): an array containing the features of
            the database images in the common space.
            - img_ids (list): the list of the image IDs of the database_images.
            - coco (MS Coco API): the API to link the img ids to MS Coco images
            - n_images (int): the number of images to retrieve
        Output:
            - list: the list of the absolute paths of the retrieved images
    '''
    # Compute the features of the tag in the textual feature space
    textual_features = sentence2vec(tag, word_model)
    # Put the tag in the common space
    common_space_features = W_text.dot(textual_features)
    # In the common space find its nearest neighbours
    idx_nearest_neigh = nearest_neighbours(
        textual_features, database_images, n_images)
    retrieved_img_ids = img_ids[idx_nearest_neigh]
    img_paths = [absolute_coco_path(img_id) for img_id in retrieved_img_ids]
    return img_paths


def image_to_tag_search(image_path, W_image, image_model, database_captions,
                        img_ids, coco, n_tags=3, expanding_factor=10):
    '''From a given tag returns the top n_images in the database closest to the
    tag in the common feature space.
        Args:
            - image_path (str): the path of the image to search
            - W_image (ndarray): the matrix of transition between the visual
            feature space to the common space.
            - image_model (keras model): the vgg 19 model
            - database_captions (ndarray): an array containing the features of
            the database tags in the common space.
            - img_ids (list): the list of the image IDs of the
            database_captions.
            - coco (MS Coco API): the API to link the img ids to MS Coco images
            - n_tags (int): the number of tags to retrieve eventually
            - expanding_factor (int): the proportion of captions we need to
            retrieve to achieve the retrieval of n_tags tags.
        Output:
            - list: the list of the retrieved tags
    '''
    # Load the image in a numpy array and process it
    img_mat = process_image(image_path)
    # Compute the features of the image in the visual feature space
    visual_features = compute_nn_features(img_mat, image_model)
    # Put the tag in the common space
    common_space_features = W_image.dot(visual_features)
    # In the common space find its nearest neighbours (we take a lot of them to
    # then select the most common tags)
    idx_nearest_neigh = nearest_neighbours(
        common_space_features, database_captions, 10*n_tags)
    img_ids = img_ids[idx_nearest_neigh]


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
    return np.argpartition(dist, k)[:, k]
