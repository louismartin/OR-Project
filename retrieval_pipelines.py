import numpy as np

from image_processing import process_image
from vgg import compute_vgg_features
from word2vec import sentence2vec


def tag_to_image_search(tag, W_text, word_model, database_images, img_ids,
                        n_images=3):
    '''From a given tag returns the top n_images in the database closest to the
    tag in the common feature space.
        Args:
            - tag (str): the tag to search
            - W_text (ndarray): the matrix of transition between the textual
            feature space to the common space.
            - word_model (gensim model): the word 2 vec model
            - database_images (ndarray): an array containing the features of
            the database images in the common space.
            - n_images (int): the number of image to retrieve
        Output:
            - list: the list of the img_ids of the retrieved images
    '''
    # Compute the features of the tag in the textual feature space
    textual_features = sentence2vec(tag)
    # Put the tag in the common space
    common_space_features = W_text.dot(textual_features)
    # In the common space find its nearest neighbours
    idx_nearest_neigh = nearest_neighbours(
        textual_features, common_space_features, n_images)
    return img_ids[idx_nearest_neigh]


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
