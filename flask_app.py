from flask import Flask, render_template, request
from joblib import load
from recommend import recommend_book

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

global FastText_model
global book_kmeans
global feelings_knn
global glove

# load in all the pre-trained models
FastText_model = load('../model/FastText_model.joblib')
books_kmeans = load('../model/book_kmeans.joblib')
feelings_knn = load('../model/feelings_knn.joblib')
glove = load('../model/glove_6B_50d.joblib')

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/recommend')
def recommend():
    feeling = request.args.get('feeling')
    recommendation_list = recommend_book(feeling, FastText_model, books_kmeans, feelings_knn, glove)

    return render_template('recommend.html', recommendation_list = recommendation_list)

if __name__ == '__main__':
    app.run(debug = True)
