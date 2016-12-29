from keras.applications import vgg19
from keras import backend as K
from keras.optimizers import SGD
import numpy as np


net = vgg19.VGG19()
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)


def compile_net():
    net.compile(optimizer=sgd, loss='categorical_crossentropy')


def compute_vgg_features(X, layer=2):
    f = K.function([net.layers[0].input, K.learning_phase()],
                   [net.layers[-layer].output])
    return f([X, 0])[0]
