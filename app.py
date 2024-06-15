from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import certifi
import os


new_df = pd.read_csv("Final_ai.csv")

app = Flask(__name__, static_url_path='/static')

# MongoDB configuration
client = MongoClient(
    "mongodb+srv://pj29102005:bTQfPPqugcyv9mv8@cluster0.9nt5ygc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    tls=True,
    tlsCAFile=certifi.where()
)
db = client['library']
books_collection = db['books_data']
feedback_collection = db['feedback']

@app.route('/')
def home():
    data = list(books_collection.find({"rating": 5}).sort("rating", 1).limit(8))
    author_names = [row['books'] for row in data]
    genre = [row['mod_title'] for row in data]
    image = [row['img'] for row in data]
    rating = [row['rating'] for row in data]
    title = [row['mod_title'] for row in data]
    
    return render_template('home.html', total_data=data,
                           author_data=author_names,
                           image_data=image,
                           title_data=title,
                           rating_data=rating,
                           genre_data=genre)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    data = []
    error = False
    if request.method == 'POST':
        title_input = request.form.get('title_input', 'None')
        print(title_input)

        cv = CountVectorizer(max_features=5000, stop_words="english")
        cv.fit_transform(new_df['books']).toarray().shape
        vectors = cv.fit_transform(new_df['books']).toarray()

        similar = cosine_similarity(vectors)
        ps = PorterStemmer()

        def stem(text):
            y = []
            for i in text.split():
                y.append(ps.stem(i))
            return " ".join(y)

        def recommend_fun(book):
            recommended_books = []
            try:
                book_index = new_df[new_df['mod_title'] == book].index[0]
                distances = similar[book_index]
                book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

                for i in book_list:
                    item = []
                    item.extend(list([new_df.iloc[i[0]].mod_title]))
                    item.extend(list([new_df.iloc[i[0]].img]))
                    item.extend(list([new_df.iloc[i[0]].rating]))
                    item.extend(list([new_df.iloc[i[0]].books]))
                    recommended_books.append(item)
                return recommended_books

            except (IndexError, KeyError) as e:
                print('\n\n', f"Exception occurred: {e}")

        data = recommend_fun(title_input)
        print('\n', "data: ", data, '\n')

        if data is None:
            error = True

    return render_template('recommend.html', data=data, error=error)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        title = request.form['title']
        author = request.form['author']
        genre = request.form['genre']
        rating = request.form['rating']
        img_url = request.form['img-url']
        
        feedback_collection.insert_one({
            'title': title,
            'author': author,
            'genre': genre,
            'img_url': img_url,
            'rating': rating
        })
        
        print("successful")
        return redirect(url_for('home'))

    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
