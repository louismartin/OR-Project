
# coding: utf-8

# # CCA on MS COCO dataset

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import time

import numpy as np
from keras.applications import vgg19
from keras.optimizers import SGD
from pycocotools.coco import COCO

from image_processing import load_images, categories, ann_file
from vgg import compute_nn_features
from text_processing import create_caption_dataframe
from word2vec import compute_textual_features


coco = COCO(ann_file)


X_visual, _, visual_img_ids = load_images(categories, coco=coco)
np.save('X_visual.npy', X_visual)


X_visual = np.load('X_visual.npy')
X_visual = X_visual[:X_visual.shape[0]//2]


net = vgg19.VGG19()
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
net.compile(optimizer=sgd, loss='categorical_crossentropy')


from tqdm import tqdm

V = np.zeros((X_visual.shape[0], 4096))
for i in tqdm(range(X_visual.shape[0]//10+1)):
    start_index = (i)*10
    end_index = (i+1)*10
    end_index = min(end_index, X_visual.shape[0])
    X_temp = X_visual[start_index:end_index]
    V_temp = compute_nn_features(X_temp, net, layer=2)
    V[start_index:end_index,:] = V_temp


np.save('V.npy', V)


df_caption = create_caption_dataframe()
T = compute_textual_features(df_caption)
textual_img_ids = df_caption.index.values


def subset_features(features, all_ids, subset_ids):
    ''' Subset features to only the samples corresponding to subset_ids
    Args:
        features (numpy array): Complete array of features as rows
        all_ids (list): list of ids corresponding to each row of features
        subset_ids (list): ids to subset textual_featurs
    '''
    features = features[np.in1d(all_ids, subset_ids)]
    return features


subset_features(T, textual_img_ids, visual_img_ids)


from sklearn.cross_decomposition import CCA

d = 128 # Dimension of the final joint latent space
cca = CCA(n_components=d, scale=False)
cca.fit(V,T)

# New basis projection matrices
W1 = cca.x_weights_
W2 = cca.y_weights_

# Compute features in the new latent space
V_latent = np.dot(V,W1)
T_latent = np.dot(T,W2)


T = np.array([[1,1],[2,2],[3,3]])
textual_image_ids = [9,15,32]
visual_image_ids = [32]
subset_features(T, textual_image_ids, visual_image_ids)

