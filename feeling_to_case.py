import numpy as np
import pandas as pd
import scipy
from joblib import load

#
def get_phrase_embedding(phrase, glove):
    '''get the GloVe embedding for a phrase, which is the avarage of all GloVe word embeddings'''
    word_embeddings = []
    for word in phrase.split(' '):
        word_embeddings.append(glove[word])
    phrase_embeddings = np.array(word_embeddings).mean(axis = 0)
    return phrase_embeddings

def feeling_to_case(feeling, feelings_knn, glove):
    '''function to match the user query to a case in The Novel Cure

    input:
    feeling: the user query string
    feelings_knn: the pre-trained k-NN model on all cases
    glove: the dictionary of pre-trained GloVe word embeddings

    output:
    the index of the case inside The Novel Cure dataset
    '''
    feeling = get_phrase_embedding(feeling, glove)
    feeling = np.reshape(feeling, (1, -1))

    distance, indices = feelings_knn.kneighbors(feeling, n_neighbors = 1)
    indices = np.reshape(indices, (-1, ))

    return indices
