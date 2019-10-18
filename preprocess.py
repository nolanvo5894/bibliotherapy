# function to remove the 'Alternate Cover Edition ...' paragraph at the beginning
def clean_cover_isbn(text):
  try:
    text = text.split('<br />')[2]
    return text
  except:
    return text
# function to remove html markings
from bs4 import BeautifulSoup
def clean_html(text):
  try:
    soup = BeautifulSoup(text, 'lxml')
    clean_text = soup.get_text()
    return clean_text
  except:
    return text
# function to remove punctuation
import string
def clean_punctuations(text):
    try:
      clean_text = "".join([c for c in text if c not in string.punctuation])
      return clean_text
    except:
      return text

def get_tokens(text):
    import nltk
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)

def clean_stop_words(text):
    from nltk.corpus import stopwords
    # nltk.download('stopwords')
    stop_words = stopwords.words('english')
    return [word for word in text if word not in stop_words]

def get_vector(text):
    from joblib import load
    import numpy as np
    model = load('FastText_model.joblib')
    vector = np.array([model.wv[word] for word in text]).mean(axis = 0)
    return vector
