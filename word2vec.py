# coding: utf-8

import re
import string
import os.path as op

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

from pycocotools.coco import COCO
from text_processing import create_caption_dataframe, sentence2vec

def compute_textual_features(df_caption, overwrite=False):
    textual_embeddings_path = op.join('dataset', 'annotations', 'textual_embeddings.npy')
    if op.exists(textual_embeddings_path) and not overwrite:
        X = np.load(textual_embeddings_path)
    else:
        ### Load Google's pre-trained Word2Vec model.
        # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
        print('\nLoading word2vec model ...')
        path = op.join('models', 'GoogleNews-vectors-negative300.bin')
        model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)


        # Compute every caption vector representation
        i = 0
        X = np.zeros((df_caption.shape[0], model.vector_size))
        for index, row in tqdm(df_caption.iterrows(), total=df_caption.shape[0],
                                desc='Computing textual embeddings'):
            X[i,:] = sentence2vec(row['caption'], model)
            i += 1
        np.save(textual_embeddings_path, X)
    return X

