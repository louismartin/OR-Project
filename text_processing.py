import re
import string
import os.path as op

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

from pycocotools.coco import COCO

# Regex to remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
# Remove punctuation and convert to set
stopwords = set([regex.sub('', word) for word in stopwords])


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
    words = [word for word in words if (word not in stopwords)]

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
