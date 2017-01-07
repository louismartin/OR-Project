from keras import backend as K
import numpy as np


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
