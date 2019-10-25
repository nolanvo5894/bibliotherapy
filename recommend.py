import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from goodreads import client
from feeling_to_case import feeling_to_case
from preprocess import preprocess
from sklearn.neighbors import NearestNeighbors

def recommend_book(feeling, FastText_model, book_kmeans, feelings_knn, glove):
    '''function to get a list of 5 books that are most relevant to the user's query

    input:
    feeling: the user query string
    FastText_model: the pre-trained FastText model on all book summaries from the Goodreads list 'Best Books Ever'
    book_kmeans: the pre-trained k-means clustering model on the vector descriptions of all book summaries from the Goodreads list 'Best Books Ever'
    feelings_knn: the pre-trained k-NN model on all cases
    glove: the dictionary of pre-trained GloVe word embeddings

    output:
    a list of 5 books that are most relevant to the user's query
    '''
    
    # create the Goodreads API client
    goodreads_key = 't6mIkabukH29jAey0381yA'
    goodreads_secret = '6IWvvO5CNFIAqUragee2Bb5HkEOvxIYqSXeXdFSHvM'
    good_client = client.GoodreadsClient(goodreads_key, goodreads_secret)

    # get the book based on the query
    cure_short = pd.read_csv('cure_short.csv')

    # get the matching case and the book prescribed for the case
    case_indices = feeling_to_case(feeling, feelings_knn, glove)

    id = cure_short.iloc[case_indices]['goodreads_id'].item()
    test_book = good_client.book(id)
    test_book_description = test_book.description
    test_book_vector = preprocess(test_book_description, FastText_model)

    # get the cluster that the prescribed book belongs to
    cluster_number = book_kmeans.predict(test_book_vector).item()
    info_books = pd.read_json('best_books_clustered.json')
    predicted_cluster = info_books[info_books['cluster'] == cluster_number]
    predicted_cluster.reset_index(drop = True, inplace = True)

    # fit a k-NN to the vector description of all the books inside the cluster
    X = list(predicted_cluster['vector_description'])
    X = np.array(X)
    books_knn = NearestNeighbors(metric = 'cosine') # should try 'mahalanobis'
    books_knn.fit(X)

    # get a list of the most similar books to the prescribed book from the cluster
    distance, indices = books_knn.kneighbors(test_book_vector, n_neighbors = 5)
    indices = indices.reshape(-1,)
    recommendation_list = list(predicted_cluster.iloc[indices]['title'])

    return recommendation_list
