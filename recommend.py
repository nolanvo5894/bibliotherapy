def recommend_book(case):
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.tokenize import word_tokenize

    from goodreads import client
    goodreads_key = 't6mIkabukH29jAey0381yA'
    goodreads_secret = '6IWvvO5CNFIAqUragee2Bb5HkEOvxIYqSXeXdFSHvM'
    good_client = client.GoodreadsClient(goodreads_key, goodreads_secret)

# get the book based on the query
    cure = pd.read_csv('cure.csv')
    id = int(cure[cure['case'] == case]['goodreads_id'].item())
    test_book = good_client.book(id)

    test_book_description = test_book.description

    from preprocess import clean_cover_isbn, clean_html, clean_punctuations, get_tokens, clean_stop_words, get_vector
    test_book_vector = clean_cover_isbn(test_book_description)
    test_book_vector = clean_html(test_book_vector)
    test_book_vector = clean_punctuations(test_book_vector)
    test_book_vector = get_tokens(test_book_vector)
    test_book_vector = clean_stop_words(test_book_vector)
    test_book_vector = get_vector(test_book_vector)
    test_book_vector = np.reshape(test_book_vector, (1, 100))

    from joblib import load
    book_kmeans = load('book_kmeans.joblib')
    cluster_number = book_kmeans.predict(test_book_vector).item()

    info_books = pd.read_json('best_books_clustered.json')
    predicted_cluster = info_books[info_books['cluster'] == cluster_number]

    predicted_cluster.reset_index(drop = True, inplace = True)

    X = list(predicted_cluster['vector_description'])
    X = np.array(X)

    from sklearn.neighbors import NearestNeighbors
    books_knn = NearestNeighbors(metric = 'cosine') # should try 'mahalanobis'
    books_knn.fit(X)

    distance, indices = books_knn.kneighbors(test_book_vector, n_neighbors = 5)
    indices = indices.reshape(-1,)
    recommendation_list = list(predicted_cluster.iloc[indices]['title'])

    return recommendation_list
