import numpy as np
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords

def clean_cover_isbn(text):
    '''function to remove the 'Alternate Cover Edition ...' paragraph at the beginning of book summaries'''
  try:
    text = text.split('<br />')[2]
    return text
  except:
    return text

def clean_html(text):
    '''function to remove html markings from the scraped book summaries'''
  try:
    soup = BeautifulSoup(text, 'lxml')
    clean_text = soup.get_text()
    return clean_text
  except:
    return text

def clean_punctuations(text):
    '''function to remove punctuations from the text'''
    try:
      clean_text = "".join([c for c in text if c not in string.punctuation])
      return clean_text
    except:
      return text

def get_tokens(text):
    '''function to tokenize the text'''
    import nltk
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)

def clean_stop_words(text):
    '''function to remove all stopwords'''
    stop_words = stopwords.words('english')
    return [word for word in text if word not in stop_words]

def get_vector(text, FastText_model):
    '''function to vectorized the text using the trained FastText_model

    input:
    text: a string
    FastText_model: the pre-trained FastText model on all book summaries from the Goodreads list 'Best Books Ever'

    output:
    the vectorized version of the text
    '''
    vector = np.array([FastText_model.wv[word] for word in text]).mean(axis = 0)
    return vector

def preprocess(text, FastText_model):
    '''the complete pipeline from raw text to vectorized text

    input:
    text: a string
    FastText_model: the pre-trained FastText model on all book summaries from the Goodreads list 'Best Books Ever'

    output:
    the vectorized version of the text
    '''
    test_book_vector = clean_cover_isbn(text)
    test_book_vector = clean_html(test_book_vector)
    test_book_vector = clean_punctuations(test_book_vector)
    test_book_vector = get_tokens(test_book_vector)
    test_book_vector = clean_stop_words(test_book_vector)
    test_book_vector = get_vector(test_book_vector, FastText_model)
    test_book_vector = np.reshape(test_book_vector, (1, 100))
    return test_book_vector
