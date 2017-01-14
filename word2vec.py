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
from tools import get_all_sorted_ids

def load_word2vec():
    # Load Google's pre-trained Word2Vec model.
    # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
    print('\nLoading word2vec model ...')
    path = op.join('models', 'GoogleNews-vectors-negative300.bin')
    model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    return model


def compute_textual_features(df_caption, overwrite=False):
    textual_embeddings_path = op.join('data', 'textual_embeddings.npy')
    if op.exists(textual_embeddings_path) and not overwrite:
        X = np.load(textual_embeddings_path)
    else:
        model = load_word2vec()

        # Compute every caption vector representation
        i = 0
        X = np.zeros((df_caption.shape[0], model.vector_size))
        for index, row in tqdm(df_caption.iterrows(),
                               total=df_caption.shape[0],
                               desc='Computing textual embeddings'):
            X[i, :] = sentence2vec(row['caption'], model)
            i += 1
        np.save(textual_embeddings_path, X)
    return X


def compute_semantic_features(coco, overwrite=False):
    '''
    Compute semantic features as an average of the categories word2vec
    representation.
    '''
    semantic_features_path = op.join('data', 'semantics_features.npy')
    semantic_ids_path = op.join('data', 'semantics_ids.npy')
    if (op.exists(semantic_features_path) and
            op.exists(semantic_ids_path) and not overwrite):
        C = np.load(semantic_features_path)
        semantic_img_ids = np.load(semantic_ids_path)
    else:
        model = load_word2vec()

        # Get all categories (sorted and no duplicates)
        cat_ids = coco.getCatIds()
        cat_ids = sorted(set(cat_ids))
        cats = coco.loadCats(cat_ids)
        cat_names = np.array([cat['name'] for cat in cats])

        img_ids = get_all_sorted_ids(coco)
        df = pd.DataFrame('', index=img_ids, columns=['categories'])

        # For each row concatenate its categroy names
        cat_ids = np.array(cat_ids)
        for cat_id in tqdm(cat_ids):
            img_ids = coco.getImgIds(catIds=cat_id)
            cat_name = cat_names[cat_ids == cat_id][0]
            df.loc[img_ids, 'categories'] += ' ' + cat_name

        # Compute every image category vector representation
        i = 0
        C = np.zeros((df.shape[0], model.vector_size))
        for index, row in tqdm(df.iterrows(),
                               total=df.shape[0],
                               desc='Computing textual embeddings'):
            C[i, :] = sentence2vec(row['categories'], model)
            i += 1
        semantic_img_ids = df.index.values
        np.save(semantic_features_path, C)
        np.save(semantic_ids_path, semantic_img_ids)
    return C, semantic_img_ids
