from keras.applications import vgg19
from keras import backend as K
from keras.optimizers import SGD
import numpy as np


net = vgg19.VGG19()
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)


def compile_net():
    """Compile net when needed
    """
    net.compile(optimizer=sgd, loss='categorical_crossentropy')


def compute_vgg_features(X, layer=2):
    """Computes vgg features of a batch of image matrices from a given layer
        Args:
            - X (ndarray: (None, 224, 224, 3)): the batch of matrices
            - layer (int): the index of the layer (starting from the output)
            you want to consider
        Output:
            - ndarray: (None, 4096): the vgg features for the batch of matrices
    """
    f = K.function([net.layers[0].input, K.learning_phase()],
                   [net.layers[-layer].output])
    return f([X, 0])[0]
