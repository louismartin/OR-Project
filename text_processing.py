import re
import string
import os.path as op

import gensim
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from tqdm import tqdm

from pycocotools.coco import COCO

# Regex to remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
# Remove punctuation and convert to set
stops = set(stopwords.words('english'))


def sentence2vec(sentence, model):
    ''' Converts a sentence to a vector by averaging each word vector
    representation given by the model.

    Args:
        sentence (str): Sentence to be converted to a vector.
        model: gensim word2vec model
    Returns:
        vec: the vector representation of sentence of size model.vector_size
    '''
    # Remove all non alphanumeric characters
    sentence = sentence.lower()
    sentence = regex.sub('', sentence)
    words = sentence.split(' ')
    words = [word for word in words if (word not in stops)]

    # Average vector representations of the words in the sentence
    vec = np.zeros(model.vector_size)
    count = 0
    for word in words:
        if word in model.vocab:
            vec += model[word]
            count += 1
    vec /= count
    return vec


def create_caption_dataframe(overwrite=False):
    '''
    Create a dataframe containing all captions for each images
    '''
    # initialize COCO api for caption annotations
    data_dir='dataset'

    dataframe_path = op.join(data_dir, 'annotations', 'captions.csv')
    if op.exists(dataframe_path) and not overwrite:
        df = pd.read_csv(dataframe_path, index_col=0)
        return df
    data_type='train2014'
    caption_file = op.join(data_dir, 'annotations', 'captions_%s.json' % data_type)
    coco = COCO(caption_file)

    # Get all image ids
    img_ids = coco.getImgIds()

    # An image can have several captions
    # Concatenate the captions for each image and save as csv
    df = pd.DataFrame(index=sorted(img_ids), columns=['caption'])
    for img_id in tqdm(img_ids, desc='Retrieving captions'):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        caption = ''
        for ann in anns:
            caption += ' ' + ann['caption']
        df.loc[img_id] = caption
    df.to_csv(dataframe_path)
    return df
