from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/recommend')
def recommend():
    feeling = request.args.get('feeling')

    import pandas as pd
    info_books = pd.read_json('best_books_clustered.json')

    from recommend import recommend_book
    recommendation_list = recommend_book(feeling)

    return render_template('recommend.html', recommendation_list = recommendation_list)

if __name__ == '__main__':
    app.run(debug = True)
