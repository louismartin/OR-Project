from image_processing import process_image
from vgg import compute_vgg_features
from word2vec import sentence2vec


def tag_to_image_search(tag, W_text, word_model, n_images=3):
    '''From a given tag returns the top n_images in the database closest to the
    tag in the common feature space.
        Args:
            - tag (str): the tag to search
            - W_text (ndarray): the matrix of transition between the textual
            feature space to the common space.
            - word_model (gensim model): the word 2 vec model
            - n_images (int): the number of image to retrieve
        Output:
            - list: the list of the img_ids of the retrieved images
    '''
    textual_features = sentence2vec(tag)
    common_space_features = W_text.dot(textual_features)
