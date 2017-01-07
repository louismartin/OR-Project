
# coding: utf-8

# Download Google's pretrained model $\href{https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing}{here}$

import re
import string
import os.path as op

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

from pycocotools.coco import COCO
from text_processing import create_caption_dataframe, sentence2vec


df_caption = create_caption_dataframe()


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
np.save('dataset/annotations/textual_embeddings.npy', X)

