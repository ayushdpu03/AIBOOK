from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from pymongo import MongoClient
import certifi
import os

app = Flask(__name__, static_url_path='/static')

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://pj29102005:bTQfPPqugcyv9mv8@cluster0.9nt5ygc.mongodb.net/library?retryWrites=true&w=majority&appName=Cluster0")
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where()
)
db = client['library']
books_collection = db['books_data']
feedback_collection = db['feedback']

# Read CSV file
try:
    new_df = pd.read_csv("Final_ai.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    new_df = pd.DataFrame()  # Use an empty dataframe if file is not found

# Initialize vectorizer and compute vectors
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df['books']).toarray()
similar = cosine_similarity(vectors)

ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

@app.route('/')
def home():
    data = list(books_collection.find({"rating": 5}).sort("rating", 1).limit(8))
    return render_template('home.html', total_data=data,
                           author_data=[row['books'] for row in data],
                           image_data=[row['img'] for row in data],
                           title_data=[row['mod_title'] for row in data],
                           rating_data=[row['rating'] for row in data],
                           genre_data=[row['mod_title'] for row in data])

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    data = []
    error = False
    if request.method == 'POST':
        title_input = request.form.get('title_input', 'None').strip()

        def recommend_fun(book):
            recommended_books = []
            try:
                book_index = new_df[new_df['mod_title'] == book].index[0]
                distances = similar[book_index]
                book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

                for i in book_list:
                    recommended_books.append([
                        new_df.iloc[i[0]].mod_title,
                        new_df.iloc[i[0]].img,
                        new_df.iloc[i[0]].rating,
                        new_df.iloc[i[0]].books
                    ])
                return recommended_books

            except (IndexError, KeyError) as e:
                print(f"Exception occurred: {e}")
                return None

        data = recommend_fun(title_input)

        if data is None:
            error = True

    return render_template('recommend.html', data=data, error=error)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback_collection.insert_one({
            'title': request.form['title'],
            'author': request.form['author'],
            'genre': request.form['genre'],
            'img_url': request.form['img-url'],
            'rating': request.form['rating']
        })
        print("successful")
        return redirect(url_for('home'))

    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
