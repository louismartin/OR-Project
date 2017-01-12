import os.path as op

from keras import backend as K
import numpy as np
from tqdm import tqdm

from image_processing import process_image

data_dir = "dataset"
data_type = "train2014"


def compute_nn_features(X, net, layer=2):
    '''Computes features of a batch of image matrices from a given layer of a
    given neural network
        Args:
            - X (ndarray: (None, 224, 224, 3)): the batch of matrices
            - net (keras model): the neural network
            - layer (int): the index of the layer (starting from the output)
            you want to consider
        Output:
            - ndarray: (None, 4096): the vgg features for the batch of matrices
    '''
    f = K.function([net.layers[0].input, K.learning_phase()],
                   [net.layers[-layer].output])
    return f([X, 0])[0]


def compute_nn_features_ids(ids, net, coco):
    V = []
    processed_ids = []
    for img_id in tqdm(ids):
        img = coco.loadImgs(img_id)[0]
        img_path = op.join(data_dir, data_type, img["file_name"])
        x = process_image(img_path)
        if len(x) > 0:
            x = np.expand_dims(x, axis=0)
            v = compute_nn_features(x, net, layer=2)[0]
            V.append(v)
            processed_ids.append(img_id)
    V = np.array(V)
    return V, processed_ids
