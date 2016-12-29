import re
import string

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

from pycocotools.coco import COCO


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
    regex = re.compile('[\W_ ]+') # TODO: compiling once for all calls might be faster
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sentence = regex.sub('', sentence)
    words = sentence.split(' ')

    # Average vector representations of the words in the sentence
    vec = np.zeros(model.vector_size)
    count = 0
    for word in words:
        if word in model.vocab:
            vec += model[word]
            count += 1
    vec /= count
    return vec

def create_caption_dataframe():
    ''' Create a dataframe containing all captions for each images '''
    # initialize COCO api for caption annotations
    dataDir='dataset'
    dataType='train2014'
    annFile = '%s/annotations/captions_%s.json'%(dataDir,dataType)
    coco = COCO(annFile)

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
    df.to_csv('%s/annotations/captions.csv'% dataDir)
    return df


df_caption = create_caption_dataframe()

# Load Google's pre-trained Word2Vec model.
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
print('\nLoading word2vec model ...')
path = './models/GoogleNews-vectors-negative300.bin'
model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)

# TODO: Might not be ideal to put vectors in a pandas dataframe
df_vec = pd.DataFrame(
    index=df_caption.index,
    columns=list(range(model.vector_size))
    )
# Compute every caption vector representation
for index, row in tqdm(df_caption.iterrows(), total=df_caption.shape[0],
                        desc='Computing textual embeddings'):
    df_vec.loc[index] = sentence2vec(row['caption'], model)
df_vec.to_csv('dataset/annotations/textual_embeddings.csv')
